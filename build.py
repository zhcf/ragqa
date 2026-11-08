"""把桌面程序打包成可分发的 Windows 目录（onedir）

用法：
    .\\env\\Scripts\\python.exe build.py             打包成无控制台窗口的正式版
    .\\env\\Scripts\\python.exe build.py --console   保留控制台，便于看报错

产物：
    dist/ragqa/
        ragqa.exe
        _internal/         Python 依赖（PySide6 / torch / transformers ...）
        llama/             llama.cpp 预编译二进制
        model/             向量模型 + GGUF 生成模型
        userguide.pdf
        ragqa使用手册.html  只讲怎么用 dist 里的程序，不涉及源码

模型和 llama 二进制不交给 PyInstaller，打完包直接复制到 exe 旁边：
既避免它对 1.5 GB 数据做分析，也让用户能随时替换手册和模型。
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_NAME = "ragqa"
DIST_DIR = BASE_DIR / "dist"
BUILD_DIR = BASE_DIR / "build"
TARGET = DIST_DIR / APP_NAME

# 打包后仍摆在 exe 旁边、可被用户替换的资源
EXTERNAL = ("llama", "model", "userguide.pdf", "ragqa使用手册.html")

# 这个项目用不到、但会被某些库顺带扫描到的东西
EXCLUDES = (
    "tkinter", "_tkinter",
    "matplotlib", "pandas", "IPython", "pytest", "notebook", "jupyter",
    "PyQt5", "PyQt6", "PySide2", "PySide6.QtWebEngineCore",
    # pymupdf.layout 在 import 时就会加载 50 MB 的 ONNX 版面模型，
    # 我们只用 get_text/get_pixmap，排除后 pymupdf 会安静地跳过它
    "pymupdf.layout", "pymupdf4llm", "pymupdf_layout",
)


def run_pyinstaller(console: bool) -> None:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onedir",
        "--name", APP_NAME,
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        "--specpath", str(BUILD_DIR),
        # sentence_transformers 用 importlib 动态加载子模块，静态分析扫不到
        "--collect-submodules", "sentence_transformers",
        "--console" if console else "--windowed",
    ]
    for name in EXCLUDES:
        cmd += ["--exclude-module", name]
    cmd.append(str(BASE_DIR / "entry.py"))

    print("PyInstaller 开始收集依赖（第一次会比较慢）…", flush=True)
    subprocess.run(cmd, check=True, cwd=str(BASE_DIR))


def copy_external() -> None:
    for name in EXTERNAL:
        src = BASE_DIR / name
        if not src.exists():
            print(f"  跳过（源不存在）：{name}")
            continue
        dst = TARGET / name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        size = (sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
                if dst.is_dir() else dst.stat().st_size)
        print(f"  已复制 {name}  {size / 1024 / 1024:.0f} MB")


def main() -> int:
    if shutil.which("pyinstaller") is None and not _has_module("PyInstaller"):
        print("缺少 PyInstaller，请先执行：pip install pyinstaller")
        return 1

    console = "--console" in sys.argv
    started = time.time()
    print(f"打包 {APP_NAME}（{'保留控制台' if console else '无控制台窗口'}）")

    run_pyinstaller(console)

    print("复制外部资源到 exe 旁边…")
    copy_external()

    exe = TARGET / f"{APP_NAME}.exe"
    total = sum(p.stat().st_size for p in TARGET.rglob("*") if p.is_file())
    print(f"\n完成，用时 {time.time() - started:.0f} 秒")
    print(f"可执行文件：{exe}")
    print(f"发布目录：  {TARGET}")
    print(f"目录大小：  {total / 1024 / 1024:.0f} MB")
    return 0


def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


if __name__ == "__main__":
    sys.exit(main())