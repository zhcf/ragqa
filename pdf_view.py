"""连续滚动的 PDF 阅读视图

一页一个 QLabel 纵向排列，滚动条滚到哪里就看到哪里，不再需要翻页按钮。

只渲染视口附近的页、离开视野的页立刻释放：106 页全渲染要占 300 多 MB，
而滚到第 80 页时其实只有那两三页需要像素。高亮画在对应页的 pixmap 上，
滚过去就能看到。
"""
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from rag_core import locate_chunk

HIGHLIGHT_FILL = QColor(255, 214, 0, 90)
HIGHLIGHT_EDGE = QColor(255, 160, 0, 200)
GAP = 12                                    # 页与页、页与边缘的间距
PAGE_STYLE = "background:#ffffff; border:1px solid #b0b0b0;"
CANVAS_STYLE = "background:#d9d9d9;"


class PdfView(QScrollArea):
    """纵向连续排列的 PDF 视图

    page_changed 在滚动到另一页时发出（1 起），界面用它更新页码。
    """

    page_changed = Signal(int)

    def __init__(self, doc, dpi=110, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.base_dpi = dpi
        self.zoom = 1.0
        self.highlight_page = None          # 0 起，高亮所在的页
        self.highlight_text = None
        self._empty = None                  # 没有手册时显示的占位提示
        self._labels = []
        self._offsets = []                  # 每页在画布中的 y 起点
        self._rendered = {}                 # 页号 → 页内高亮偏移，键存在即表示已渲染
        self._range = None                  # 上次处理的可见范围，避免重复渲染
        self._current = -1

        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        self.canvas = QWidget()
        self.canvas.setStyleSheet(CANVAS_STYLE)
        self.box = QVBoxLayout(self.canvas)
        self.box.setContentsMargins(GAP, GAP, GAP, GAP)
        self.box.setSpacing(GAP)
        self.setWidget(self.canvas)

        self._build()
        self.verticalScrollBar().valueChanged.connect(self._update_visible)

    # ================== 搭建与尺寸 ==================
    def _build(self):
        if self.doc is None:
            self._show_empty()
            return
        for index in range(self.doc.page_count):
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(PAGE_STYLE)
            self.box.addWidget(label, 0, Qt.AlignHCenter | Qt.AlignTop)
            self._labels.append(label)
        self._relayout()

    def _show_empty(self):
        """没有手册时的占位：发布出去的程序第一次打开就是这个样子"""
        self._empty = QLabel("还没有打开手册\n\n点上方「打开 PDF…」选一本 PDF")
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setFixedSize(560, 720)
        self._empty.setStyleSheet(
            "background:#ffffff; border:1px dashed #b0b0b0; color:#888; font-size:15px;")
        self.box.addWidget(self._empty, 0, Qt.AlignHCenter | Qt.AlignTop)
        self.canvas.setMinimumSize(560 + 2 * GAP, 720 + 2 * GAP)

    def _page_size(self, index):
        rect = self.doc[index].rect
        scale = self.base_dpi * self.zoom / 72.0
        return max(1, int(rect.width * scale)), max(1, int(rect.height * scale))

    def _relayout(self):
        """按当前缩放重设每页尺寸，并记下每页在画布里的纵向起点"""
        self._offsets = []
        y = GAP
        widest = 0
        for index, label in enumerate(self._labels):
            width, height = self._page_size(index)
            label.setFixedSize(width, height)
            label.setText(f"第 {index + 1} 页")          # 渲染前的占位
            widest = max(widest, width)
            self._offsets.append(y)
            y += height + GAP
        self.canvas.setMinimumSize(widest + 2 * GAP, y)
        self._range = None

    # ================== 渲染 ==================
    def _render(self, index, force=False):
        if index in self._rendered and not force:
            return
        page = self.doc[index]
        pix = page.get_pixmap(dpi=int(self.base_dpi * self.zoom))
        fmt = QImage.Format_RGB888 if pix.n == 3 else QImage.Format_RGBA8888
        image = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)
        pixmap = QPixmap.fromImage(image.copy())      # samples 会被释放，必须拷贝

        anchor = None
        if index == self.highlight_page and self.highlight_text:
            scale = pixmap.width() / page.rect.width
            rects = locate_chunk(page, self.highlight_text)
            if rects:
                painter = QPainter(pixmap)
                painter.setBrush(HIGHLIGHT_FILL)
                painter.setPen(QPen(HIGHLIGHT_EDGE, 1))
                for rect in rects:
                    painter.drawRect(QRectF(rect.x0 * scale, rect.y0 * scale,
                                            rect.width * scale, rect.height * scale))
                painter.end()
                anchor = max(0, int(rects[0].y0 * scale) - 80)

        label = self._labels[index]
        label.setText("")
        label.setPixmap(pixmap)
        self._rendered[index] = anchor

    def _release_outside(self, first, last):
        """把视野外的页换成占位，释放像素内存"""
        for index in list(self._rendered):
            if first <= index <= last:
                continue
            label = self._labels[index]
            label.clear()
            label.setText(f"第 {index + 1} 页")
            del self._rendered[index]

    # ================== 可见范围 ==================
    def _visible_range(self):
        top = self.verticalScrollBar().value()
        bottom = top + self.viewport().height()
        first = last = None
        for index, y in enumerate(self._offsets):
            height = self._labels[index].height()
            if y + height < top:
                continue
            if y > bottom:
                break
            if first is None:
                first = index
            last = index
        if first is None:
            return 0, 0
        return max(0, first - 1), min(len(self._labels) - 1, last + 1)

    def _update_visible(self):
        if not self._offsets:
            return
        first, last = self._visible_range()
        if (first, last) != self._range:
            self._range = (first, last)
            for index in range(first, last + 1):
                self._render(index)
            self._release_outside(first, last)

        current = self.current_page()
        if current != self._current:
            self._current = current
            self.page_changed.emit(current + 1)

    def current_page(self):
        """按视口上三分之一处判断当前页，滚动时切页更贴近人的感觉"""
        top = self.verticalScrollBar().value() + self.viewport().height() // 3
        for index in range(len(self._offsets) - 1, -1, -1):
            if self._offsets[index] <= top:
                return index
        return 0

    # ================== 对外接口 ==================
    def set_document(self, doc):
        """换一份 PDF：拆掉旧页，按新文档重新搭一遍

        只负责界面，旧 doc 由调用方关闭；此时旧页的 pixmap 已随控件释放。
        """
        for label in self._labels:
            self.box.removeWidget(label)
            label.deleteLater()
        if self._empty is not None:
            self.box.removeWidget(self._empty)
            self._empty.deleteLater()
            self._empty = None
        self._labels = []
        self._offsets = []
        self._rendered.clear()
        self._range = None
        self._current = -1
        self.highlight_page = None
        self.highlight_text = None
        self.doc = doc
        self._build()
        self.verticalScrollBar().setValue(0)
        self._update_visible()

    def goto_page(self, page_no, highlight_text=None):
        """跳到指定页；给了 highlight_text 就在该页标黄并滚到高亮处"""
        count = len(self._labels)
        if count == 0:                             # 还没有手册，没有页可跳
            return
        index = max(0, min(count - 1, page_no - 1))
        old = self.highlight_page
        if highlight_text is not None:
            self.highlight_page = index
            self.highlight_text = highlight_text
            if old is not None and old != index and old in self._rendered:
                self._render(old, force=True)         # 重绘旧页，抹掉上次的高亮

        self._render(index, force=highlight_text is not None)
        target = self._offsets[index]
        anchor = self._rendered.get(index)
        if highlight_text is not None and anchor is not None:
            target += anchor
        self.verticalScrollBar().setValue(target)
        self._update_visible()

    def set_zoom(self, value):
        value = max(0.5, min(2.5, round(value, 2)))
        if self.doc is None or abs(value - self.zoom) < 1e-6:
            return
        anchor = self.current_page()
        self.zoom = value
        for index in list(self._rendered):
            self._labels[index].clear()
        self._rendered.clear()
        self._relayout()
        self._update_visible()
        self.verticalScrollBar().setValue(self._offsets[anchor])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_visible()