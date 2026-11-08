"""集中配置：模型清单、路径、检索与生成参数

要新增可切换的模型，只需在 MODELS 里加一条，并把对应的 .gguf 放进 model/gguf/。
界面上的下拉框会自动读出这些条目。
"""
import os
import sys
from pathlib import Path

# 打包成 exe 后 __file__ 指向程序内部（_internal），
# 而 llama/、model/、userguide.pdf 都放在 exe 旁边，所以按 exe 定位。
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

# ====================== 路径 ======================
PDF_PATH = BASE_DIR / "userguide.pdf"              # 随包附带的示例手册，启动不自动打开
EMBED_PATH = BASE_DIR / "model" / "bge-small-zh-v1.5"
GGUF_DIR = BASE_DIR / "model" / "gguf"
LLAMA_SERVER = BASE_DIR / "llama" / "llama-server.exe"

# ====================== 生成模型 ======================
# 默认 3.5-4B：2B 能答出“念得出来”的条目，但需要归纳、对比、组织多句时
# 经常漏要点（更早的 Qwen2.5-1.5B 问题更明显，已移除）。
# Qwen3.5 小模型默认关闭思考（不浪费 CPU 生成思考过程），
# CPU 上 4B 约 4～8 秒/问，是质量与速度的平衡点。
# 缺 .gguf 的条目会在下拉框里显示「（未下载）」并置灰，不影响启动。
DEFAULT_MODEL = "qwen3.5-4b"

MODELS = {
    "qwen3.5-2b": {
        "label": "Qwen3.5-2B",
        "gguf": GGUF_DIR / "Qwen3.5-2B-Q4_K_M.gguf",
    },
    "qwen3.5-4b": {
        "label": "Qwen3.5-4B",
        "gguf": GGUF_DIR / "Qwen3.5-4B-Q4_K_M.gguf",
    },
    "qwen3.5-9b": {
        "label": "Qwen3.5-9B",
        "gguf": GGUF_DIR / "Qwen3.5-9B-Q4_K_M.gguf",
    },
}

# ====================== llama-server 参数 ======================
LLM_HOST = "127.0.0.1"
# 避开 8080/8081 这类常被其他开发工具占用的端口；真的被占了会自动换一个空闲的
LLM_PORT = 18321
LLM_N_CTX = 4096
LLM_THREADS = min(8, os.cpu_count() or 8)
MAX_NEW_TOKENS = 300
LLM_TIMEOUT = 300          # 单次生成的最长等待秒数
LLM_START_TIMEOUT = 180    # 冷启动加载模型的最长等待秒数

# ====================== 检索参数 ======================
TOP_K = 4
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
# 法规条目超过这个长度就再按 CHUNK_SIZE 切：附则、处罚条款动辄上千字，
# 整条当一块会让检索颗粒太粗、prompt 凭空胀大。普通文档不受影响。
MAX_ARTICLE_LEN = 800
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# ====================== 界面参数 ======================
PDF_DPI = 110              # PDF 渲染分辨率，越大越清晰也越占内存