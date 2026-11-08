"""PDF 手册问答桌面程序

启动时只在记住过手册的情况下解析它并建索引，然后拉起 llama-server；
第一次打开（没有记录）时 PDF 区留空，等用户自己选一本。
工具栏可以随时换一本 PDF（换完自动重建索引），也可以切换生成模型（只重启 llama-server）。
"""
import atexit
import html
import os
import re
import sys
import time
from pathlib import Path

import pymupdf
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton,
                               QSplitter, QTextBrowser, QToolBar, QVBoxLayout,
                               QWidget)

from config import BASE_DIR, DEFAULT_MODEL, MODELS, PDF_DPI
from llama_client import LlamaServer
from pdf_view import PdfView
from rag_core import DocIndex, answer_question

PID_FILE = BASE_DIR / ".app.pid"
LOG_FILE = BASE_DIR / "app.log"
SETTINGS_FILE = BASE_DIR / "settings.ini"      # 记住上次的模型和手册，删掉即恢复默认


def ensure_streams():
    """pythonw 没有控制台，sys.stdout/stderr 是 None

    sentence-transformers / tqdm 加载时会往 stderr 写进度条，遇到 None 会抛
    AttributeError，表现为“索引失败”。这里换成空设备。
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8", errors="replace"))


def log_error(message):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except OSError:
        pass


def load_settings():
    """读出 settings.ini 的所有键值，文件不存在时返回空字典"""
    try:
        text = SETTINGS_FILE.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # 记事本另存成 ANSI 时解码会失败，按“没有设置”处理，别挡住启动
        return {}
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def save_settings(**changes):
    """记住这次的选择；目录只读时写不进去，不影响使用

    每次都先读再合并：模型和手册各改各的，不能把对方抹掉。
    """
    data = load_settings()
    data.update({key: value for key, value in changes.items() if value})
    try:
        body = "".join(f"{key}={value}\n" for key, value in data.items())
        SETTINGS_FILE.write_text(
            "# ragqa 记住的用户选择，删掉本文件即恢复默认\n" + body,
            encoding="utf-8",
        )
    except OSError:
        pass


def resolve_initial_model():
    """启动用哪个模型：上次的选择 -> DEFAULT_MODEL -> 任何一个能用的

    模型文件可能被删掉腾空间，所以每一级都要确认 GGUF 还在，
    否则会把一个不存在的模型塞进下拉框，启动阶段直接报错。
    """
    for key in (load_settings().get("model"), DEFAULT_MODEL):
        if key in MODELS and Path(MODELS[key]["gguf"]).exists():
            return key
    for key, spec in MODELS.items():
        if Path(spec["gguf"]).exists():
            return key
    return DEFAULT_MODEL                  # 一个都不在，交给启动流程报错提示


def resolve_initial_pdf():
    """启动读哪本手册：上次选的那本；没有记录或打不开就返回 None

    发布给别人的程序第一次打开时 PDF 区应该是空的，不能自动塞一本手册；
    记录的文件也可能被挪走、删掉，所以先真打开一次确认能用，
    坏路径不能让程序每次启动都失败。
    """
    saved = load_settings().get("pdf")
    if not saved:
        return None
    path = Path(saved)
    if not path.is_file():
        return None
    try:
        doc = pymupdf.open(str(path))
    except Exception:
        return None
    try:
        usable = doc.page_count > 0 and not doc.needs_pass
    finally:
        doc.close()
    return path if usable else None


class Worker(QThread):
    """把耗时操作丢到后台线程，结果通过信号回主线程"""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self):
        try:
            self.done.emit(self._fn(*self._args, **self._kwargs))
        except Exception as exc:                       # 后台线程里的异常不能吞掉
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class QuestionEdit(QPlainTextEdit):
    """多行提问框：Enter 直接提交，Shift+Enter 换行"""

    submitted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("输入问题，例如：如何给作业添加标签（Shift+Enter 换行）")
        self.setFixedHeight(78)                        # 大约三行，写长点也不用拖着看
        self.setTabChangesFocus(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)           # Shift+Enter 换行
            else:
                self.submitted.emit()
            return
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF 手册问答 · 本地 llama.cpp")
        self.resize(1300, 880)

        self.server = LlamaServer()
        self.index = None
        self.pdf_path = resolve_initial_pdf()
        self.pdf_doc = pymupdf.open(str(self.pdf_path)) if self.pdf_path else None
        self.workers = []
        self.busy = False
        self.active_model = resolve_initial_model()

        self._build_ui()
        self._bootstrap()

    # ================== 界面搭建 ==================
    def _build_ui(self):
        bar = QToolBar("主工具栏")
        bar.setMovable(False)
        self.addToolBar(bar)

        bar.addWidget(QLabel("  生成模型："))
        self.model_box = QComboBox()
        self.model_box.setMinimumWidth(220)
        for key, spec in MODELS.items():
            self.model_box.addItem(spec["label"], key)
            if not Path(spec["gguf"]).exists():
                row = self.model_box.count() - 1
                self.model_box.model().item(row).setEnabled(False)
                self.model_box.setItemText(row, spec["label"] + "（未下载）")
        self.model_box.setCurrentIndex(list(MODELS).index(self.active_model))
        self.model_box.activated.connect(self._on_model_changed)
        bar.addWidget(self.model_box)

        bar.addSeparator()
        self.open_btn = QPushButton("打开 PDF…")
        self.open_btn.setToolTip("换一本要问答的手册，换完会自动重建索引")
        self.open_btn.clicked.connect(self._on_open_pdf)
        bar.addWidget(self.open_btn)

        self.rebuild_btn = QPushButton("重建索引")
        self.rebuild_btn.clicked.connect(self._on_rebuild)
        bar.addWidget(self.rebuild_btn)

        bar.addSeparator()
        self.status_label = QLabel()
        bar.addWidget(self.status_label)
        self.set_status("正在启动…", "#888")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_ask_panel())
        splitter.addWidget(self._build_pdf_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([520, 780])
        self.setCentralWidget(splitter)

    def _build_ask_panel(self):
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(10, 10, 10, 10)
        box.setSpacing(8)

        row = QHBoxLayout()
        self.question_edit = QuestionEdit()
        self.question_edit.submitted.connect(self._on_ask)
        self.ask_btn = QPushButton("提问")
        self.ask_btn.setMinimumWidth(80)
        self.ask_btn.setMinimumHeight(32)
        self.ask_btn.clicked.connect(self._on_ask)
        row.addWidget(self.question_edit, 1)
        row.addWidget(self.ask_btn, 0, Qt.AlignBottom)
        box.addLayout(row)

        self.answer_view = QTextBrowser()
        self.answer_view.setOpenLinks(False)
        self.answer_view.anchorClicked.connect(self._on_anchor)
        self.answer_view.setStyleSheet("QTextBrowser{font-size:14px;}")
        box.addWidget(self.answer_view, 3)

        box.addWidget(QLabel("检索依据（点一条可跳到对应页并高亮）"))
        self.hits_view = QListWidget()
        self.hits_view.itemClicked.connect(self._on_hit_clicked)
        self.hits_view.setStyleSheet("QListWidget{font-size:13px;}")
        box.addWidget(self.hits_view, 2)

        return panel

    def _build_pdf_panel(self):
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)

        self.pdf_view = PdfView(self.pdf_doc, PDF_DPI)
        self.pdf_view.page_changed.connect(self._on_page_changed)

        nav = QHBoxLayout()
        nav.setContentsMargins(8, 6, 8, 0)
        self.file_label = QLabel()
        self.file_label.setStyleSheet("color:#333;font-weight:600;")
        self._set_pdf_title(self.pdf_path)
        self.page_label = QLabel(
            f"1 / {self.pdf_doc.page_count}" if self.pdf_doc is not None else "—")
        hint = QLabel("滚动即翻页")
        hint.setStyleSheet("color:#999;")
        zoom_out = QPushButton("缩小")
        zoom_out.clicked.connect(lambda: self.pdf_view.set_zoom(self.pdf_view.zoom - 0.15))
        zoom_in = QPushButton("放大")
        zoom_in.clicked.connect(lambda: self.pdf_view.set_zoom(self.pdf_view.zoom + 0.15))
        nav.addWidget(self.file_label)
        nav.addSpacing(12)
        nav.addWidget(self.page_label)
        nav.addWidget(hint)
        nav.addStretch(1)
        nav.addWidget(zoom_out)
        nav.addWidget(zoom_in)
        box.addLayout(nav)
        box.addWidget(self.pdf_view, 1)
        return panel

    # ================== 启动流程 ==================
    def _bootstrap(self):
        self.busy = True
        self._refresh_controls()
        if self.pdf_doc is None:
            # 第一次打开没有手册：PDF 区留空，先把生成模型拉起来，选完手册就能马上问
            self._start_model()
            return
        self.set_status("正在加载向量模型并建立索引…", "#c07f00")
        self._run(self._build_index, on_done=self._on_index_ready,
                  on_fail=self._on_index_failed)

    def _build_index(self):
        return DocIndex(self.pdf_path)

    def _on_index_ready(self, index):
        self.index = index
        self.goto_page(1)
        self._start_model()

    def _start_model(self):
        """拉起 llama-server

        通常排在索引之后：向量模型和 GGUF 模型同时加载会抢内存和 IO，
        串行还不到 8 秒，但状态提示更明确、出问题也更好定位。
        第一次打开没有手册时没有索引可建，就直接启动它。
        """
        label = MODELS[self.active_model]["label"]
        self.set_status(f"正在启动 llama-server（{label}）…", "#c07f00")
        self._run(self.server.start, self.active_model,
                  on_done=self._on_model_ready, on_fail=self._on_model_failed)

    def _on_index_failed(self, message):
        log_error(f"index failed: {message}")
        self.set_status("索引失败", "#c62828")
        QMessageBox.critical(self, "建立索引失败", message)
        self.busy = False
        self._refresh_controls()

    def _on_model_ready(self, _seconds):
        save_settings(model=self.active_model)   # 首次启动和切换成功都会落到这里
        self._maybe_ready()

    def _on_model_failed(self, message):
        log_error(f"model failed: {message}")
        self.set_status("模型启动失败", "#c62828")
        QMessageBox.critical(self, "生成模型启动失败", message)
        self.busy = False                  # 让用户能重新选择模型
        self._refresh_controls()

    def _maybe_ready(self):
        """索引与生成模型都就绪时才解开 busy；还没有手册时，模型就绪就够了"""
        if not self.server.is_ready():
            return
        if self.index is None and self.pdf_doc is not None:
            return                         # 有手册但索引还没建好，继续等
        self.busy = False
        if self.index is None:
            self.set_status("请点「打开 PDF…」选择一本手册", "#c07f00")
        else:
            label = MODELS[self.active_model]["label"]
            self.set_status(
                f"就绪 · {label} · {len(self.index.chunks)} 块 / {self.index.num_pages} 页",
                "#2e7d32",
            )
        self._refresh_controls()

    # ================== 换手册 ==================
    def _on_open_pdf(self):
        if self.busy:
            QMessageBox.information(self, "请稍候", "当前任务还没结束，稍后再换手册。")
            return
        name, _ = QFileDialog.getOpenFileName(
            self, "选择 PDF 手册",
            str(self.pdf_path.parent) if self.pdf_path else str(BASE_DIR),
            "PDF 文件 (*.pdf)")
        if name:
            self._switch_pdf(Path(name))

    def _switch_pdf(self, path):
        """真正换手册；打不开时保持原样并返回 False

        先确认新文件能打开再关旧的，顺序反了就成了「旧的关了、新的打不开」；
        旧文档放在 finally 里关，中途出错也不会把它漏掉。
        """
        if path == self.pdf_path:
            return False

        try:                                   # 先确认能打开，打不开就别动现有状态
            doc = pymupdf.open(str(path))
        except Exception as exc:
            QMessageBox.critical(self, "打不开这个文件", f"{type(exc).__name__}: {exc}")
            return False
        if doc.page_count == 0:
            doc.close()
            QMessageBox.critical(self, "打不开这个文件", "这个 PDF 里没有页面。")
            return False
        if doc.needs_pass:                     # 加密 PDF：open() 不报错，翻页才报，得提前拦
            doc.close()
            QMessageBox.critical(self, "打不开这个文件", "这个 PDF 需要密码，暂时打不开。")
            return False

        old = self.pdf_doc
        self.pdf_doc = doc
        self.pdf_path = path
        self.index = None                      # 旧索引属于旧文档，先作废
        self.answer_view.clear()
        self.hits_view.clear()
        try:
            self.pdf_view.set_document(doc)
            self._set_pdf_title(path)
            self.page_label.setText(f"1 / {doc.page_count}")
        finally:
            # 换页本身也可能出错，旧文档无论如何都要关，否则文件在 Windows 上一直被锁着
            if old is not None:
                try:
                    old.close()
                except Exception as exc:
                    log_error(f"close old pdf failed: {exc}")

        self.busy = True
        self._refresh_controls()
        self.set_status(f"正在为「{path.name}」建索引…", "#c07f00")
        self._run(self._build_index, on_done=self._on_pdf_indexed,
                  on_fail=self._on_index_failed, finish=self._clear_busy)
        return True

    def _on_pdf_indexed(self, index):
        self.index = index
        self.goto_page(1)
        # 索引真的建好才算换书成功：扫描件在这里失败，不该被记住
        save_settings(pdf=str(self.pdf_path))
        if self.server.is_ready():
            self._maybe_ready()
        else:
            # 第一次打开时模型可能还在启动，等 _on_model_ready 再显示就绪
            self.set_status("索引已建好，正在等生成模型启动…", "#c07f00")

    # ================== 模型切换 ==================
    def _on_model_changed(self, row):
        key = self.model_box.itemData(row)
        if not key or key == self.active_model:
            return
        if self.busy:
            QMessageBox.information(self, "请稍候", "当前任务还没结束，稍后再切换模型。")
            self.model_box.setCurrentIndex(list(MODELS).index(self.active_model))
            return

        previous = self.active_model
        self.active_model = key
        self.busy = True
        self._refresh_controls()
        self.set_status(f"正在加载 {MODELS[key]['label']}…", "#c07f00")
        self._run(self.server.start, key, on_done=self._on_model_ready,
                  on_fail=lambda message: self._on_switch_failed(previous, message),
                  finish=self._clear_busy)

    def _on_switch_failed(self, previous, message):
        """切换失败时把下拉框退回原来的选择，否则界面显示的模型和服务实际状态对不上"""
        log_error(f"switch failed: {message}")
        self.active_model = previous
        self.model_box.setCurrentIndex(list(MODELS).index(previous))
        self.set_status("切换失败", "#c62828")
        QMessageBox.critical(self, "切换模型失败",
                            f"{message}\n\n可以再选一次，或重新运行 start.bat。")

    def _on_rebuild(self):
        if self.busy:
            return
        self.busy = True
        self._refresh_controls()
        self.rebuild_btn.setEnabled(False)
        self.set_status("正在重建索引…", "#c07f00")
        self._run(self._build_index, on_done=self._on_rebuild_done,
                  on_fail=self._on_index_failed, finish=self._clear_busy)

    def _on_rebuild_done(self, index):
        self.index = index
        self.set_status(f"索引已重建 · {len(index.chunks)} 块", "#2e7d32")

    # ================== 提问 ==================
    def _on_ask(self):
        question = self.question_edit.toPlainText().strip()
        if not question:
            return
        if not self.can_ask():
            QMessageBox.information(self, "还不能提问", "索引或生成模型尚未就绪。")
            return

        self.busy = True
        self._refresh_controls()
        self.answer_view.setHtml("<p style='color:#888'>正在检索并生成答案…</p>")
        self.set_status("正在生成答案…", "#c07f00")
        self._run(answer_question, question, self.index, self.server,
                  on_done=self._on_answer, on_fail=self._on_ask_failed,
                  finish=self._clear_busy)

    def _on_answer(self, result):
        answer, hits, mode, elapsed, _usage = result
        self.show_answer(answer, mode, elapsed)
        self.hits_view.clear()
        if not hits:
            # 模型明确回答“没找到”时不给检索块，提示一句，免得空列表像出错了
            tip = QListWidgetItem("资料里没有与这个问题相关的段落")
            tip.setFlags(Qt.NoItemFlags)
            self.hits_view.addItem(tip)
        for hit in hits:
            item = QListWidgetItem(f"第 {hit['page']} 页 · 相似度 {hit['score']:.3f} · {hit['text'][:34]}…")
            item.setData(Qt.UserRole, hit)
            self.hits_view.addItem(item)
        if hits:
            # highlight 是块里跟问题最相关的几句，拿它定位才不会标在页面顶上
            self.goto_page(hits[0]["page"], hits[0].get("highlight") or hits[0]["text"])
        self.set_status(f"完成 · {mode} · {elapsed}s", "#2e7d32")

    def _on_ask_failed(self, message):
        log_error(f"ask failed: {message}")
        self.answer_view.setHtml(f"<p style='color:#c62828'>出错了：{html.escape(message)}</p>")
        self.set_status("生成失败", "#c62828")

    def show_answer(self, text, mode, elapsed):
        body = html.escape(text)
        body = re.sub(r"第\s*(\d+)\s*页", r'<a href="page:\1">第 \1 页</a>', body)
        body = body.replace("\n", "<br>")
        tag = {"llm": "生成模型", "none": "未找到相关内容"}.get(
            mode, "抽取式兜底（生成模型未给出有效答案）")
        self.answer_view.setHtml(
            f"<div style='line-height:1.75'>{body}</div>"
            f"<hr><p style='color:#888;font-size:12px'>来源：{tag} · 耗时 {elapsed}s</p>"
        )

    # ================== PDF 显示 ==================
    def _set_pdf_title(self, path):
        """右上角显示手册名，太长就从中间省略"""
        if path is None:
            self.file_label.setText("未选择手册")
            self.file_label.setToolTip("")
            return
        metrics = self.file_label.fontMetrics()
        self.file_label.setText(metrics.elidedText(path.name, Qt.ElideMiddle, 260))
        self.file_label.setToolTip(str(path))

    def goto_page(self, page_no, highlight_text=None):
        self.pdf_view.goto_page(page_no, highlight_text)

    def _on_page_changed(self, page_no):
        if self.pdf_doc is None:
            return
        self.page_label.setText(f"{page_no} / {self.pdf_doc.page_count}")

    # ================== 交互小工具 ==================
    def _on_anchor(self, url):
        target = url.toString()
        if target.startswith("page:"):
            self.goto_page(int(target[5:]))

    def _on_hit_clicked(self, item):
        hit = item.data(Qt.UserRole)
        self.goto_page(hit["page"], hit.get("highlight") or hit["text"])

    def can_ask(self):
        return not self.busy and self.index is not None and self.server.is_ready()

    def _refresh_controls(self):
        self.ask_btn.setEnabled(self.can_ask())
        self.question_edit.setEnabled(self.can_ask())
        self.model_box.setEnabled(not self.busy)
        self.open_btn.setEnabled(not self.busy)
        self.rebuild_btn.setEnabled(not self.busy and self.index is not None)

    def _clear_busy(self):
        self.busy = False
        self._refresh_controls()

    def set_status(self, text, color="#555"):
        self.status_label.setText(f'<span style="color:{color}">●</span> {html.escape(text)}')

    def _run(self, fn, *args, on_done, on_fail, finish=None):
        worker = Worker(fn, *args)
        self.workers.append(worker)

        def cleanup():
            if worker in self.workers:
                self.workers.remove(worker)
            if finish is not None:
                finish()

        worker.done.connect(on_done)
        worker.failed.connect(on_fail)
        worker.finished.connect(cleanup)
        worker.start()

    # ================== 收尾 ==================
    def closeEvent(self, event):
        if self.busy:
            reply = QMessageBox.question(self, "退出", "任务还在进行，确定退出？")
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self):
        self.server.stop()
        try:
            if self.pdf_doc is not None:
                self.pdf_doc.close()
        except Exception:
            pass
        if PID_FILE.exists():
            try:
                PID_FILE.unlink()
            except OSError:
                pass


def main():
    ensure_streams()
    try:
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass                                   # 装在只读目录时不影响使用
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
    except Exception as exc:                    # pythonw 下没有控制台，异常要弹窗
        QMessageBox.critical(None, "启动失败", f"{type(exc).__name__}: {exc}")
        if PID_FILE.exists():
            PID_FILE.unlink()
        return 1
    atexit.register(window.shutdown)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())