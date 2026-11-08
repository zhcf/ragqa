"""打包入口

单独隔一层是为了接住「import 阶段」的异常：--windowed 模式下没有控制台，
未捕获的异常只会弹一个一闪而过的对话框。这里把 traceback 落到 exe 旁边的
startup-error.log，用户和我们都能查到原因。

源码运行仍用 desktop.py，这个文件只在打包时作为入口。
"""
import sys
import traceback
from pathlib import Path


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


try:
    import desktop
except Exception:
    try:
        (_base_dir() / "startup-error.log").write_text(
            traceback.format_exc(), encoding="utf-8")
    except OSError:
        pass
    raise

if __name__ == "__main__":
    sys.exit(desktop.main())