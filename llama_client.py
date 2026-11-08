"""llama-server 进程管理

用 llama.cpp 官方预编译的 llama-server.exe 提供 OpenAI 兼容接口：
  POST /v1/chat/completions
切换模型 = 结束旧进程、带着新 GGUF 重新启动。GGUF 用 mmap 加载，实测冷启动只需 1 秒左右。
"""
import socket
import subprocess
import time
from pathlib import Path

import psutil
import requests

from config import (BASE_DIR, LLAMA_SERVER, LLM_HOST, LLM_N_CTX, LLM_PORT,
                    LLM_START_TIMEOUT, LLM_THREADS, LLM_TIMEOUT,
                    MAX_NEW_TOKENS, MODELS)

# 不弹黑色控制台窗口
CREATE_NO_WINDOW = 0x08000000


class LlamaServer:
    """一个 llama-server 子进程的生命周期与调用封装"""

    def __init__(self, log_path=None):
        self.host = LLM_HOST
        self.port = LLM_PORT
        self.proc = None
        self.model_key = None
        self.log_path = Path(log_path) if log_path else BASE_DIR / "llama-server.log"
        self._log_file = None

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    # ---------- 状态 ----------
    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def is_ready(self):
        return self.is_running() and self.model_key is not None

    def _port_open(self, timeout=0.3):
        with socket.socket() as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((self.host, self.port)) == 0

    # ---------- 启动 / 切换 ----------
    def start(self, model_key, attempts=2):
        """加载指定模型。已经加载同一个模型时直接返回 0 秒。

        个别环境里创建子进程会偶发 STATUS_DLL_INIT_FAILED（0xC0000142）：
        进程秒退、日志一个字都没写。这种失败与模型无关，重试一次通常就正常，
        所以只对「进程自己退出」做重试，超时不重试（那多半是模型太大或机器太忙）。
        """
        if model_key not in MODELS:
            raise KeyError(f"未知模型：{model_key}")
        gguf = Path(MODELS[model_key]["gguf"])
        if not gguf.exists():
            raise FileNotFoundError(f"模型文件不存在：{gguf}")
        if not LLAMA_SERVER.exists():
            raise FileNotFoundError(f"找不到 llama-server：{LLAMA_SERVER}")

        if self.is_running() and self.model_key == model_key:
            return 0.0

        last_error = None
        for attempt in range(attempts):
            try:
                return self._spawn(model_key, gguf)
            except RuntimeError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(0.5)
        raise last_error

    def _spawn(self, model_key, gguf):
        self.stop()
        self._free_port()

        cmd = [
            str(LLAMA_SERVER),
            "-m", str(gguf),
            "-c", str(LLM_N_CTX),
            "-t", str(LLM_THREADS),
            "--host", self.host,
            "--port", str(self.port),
            "--no-webui",
        ]
        # llama-server 自己会写不少日志，落到文件里方便排查
        self._log_file = open(self.log_path, "ab")
        try:
            # stdin 必须显式给：pythonw 下没有控制台，继承过来的句柄无效，
            # CreateProcess 会报 WinError 5（拒绝访问）
            self.proc = subprocess.Popen(
                cmd, cwd=str(BASE_DIR),
                stdin=subprocess.DEVNULL,
                stdout=self._log_file, stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
            )
        except Exception:
            self._log_file.close()
            self._log_file = None
            raise

        started = time.time()
        while time.time() - started < LLM_START_TIMEOUT:
            if self.proc.poll() is not None:
                code = self.proc.returncode
                self.stop()
                raise RuntimeError(
                    f"llama-server 启动失败（退出码 {code}），详见 {self.log_path.name}"
                )
            try:
                if requests.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                    self.model_key = model_key
                    return time.time() - started
            except requests.RequestException:
                pass
            time.sleep(0.2)

        self.stop()
        raise TimeoutError(f"等待 llama-server 就绪超过 {LLM_START_TIMEOUT} 秒")

    def stop(self):
        """结束子进程，可重复调用"""
        proc, self.proc = self.proc, None
        self.model_key = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

    def _free_port(self):
        """腾出监听端口

        上一轮异常退出可能留下占着端口的 llama-server，先结束它；
        如果是别的程序占着（开发机上很常见），就换一个空闲端口。
        """
        if not self._port_open():
            return
        for proc in psutil.process_iter(["name"]):
            name = (proc.info.get("name") or "").lower()
            if name.startswith("llama-server"):
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except psutil.Error:
                    pass
        for _ in range(30):
            if not self._port_open():
                return
            time.sleep(0.1)

        with socket.socket() as sock:
            sock.bind((self.host, 0))
            self.port = sock.getsockname()[1]

    # ---------- 调用 ----------
    def chat(self, messages, max_tokens=MAX_NEW_TOKENS, temperature=0.0):
        """返回 (回答文本, usage 字典)"""
        if not self.is_ready():
            raise RuntimeError("生成模型尚未就绪")
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            # Qwen3/3.5 的 chat template 默认开思考，CPU 上整段 token 预算会花在
            # 思考过程里，正文还没开始就被截断（实测 300 token 全空）。传 false
            # 直接出答案；Qwen2.5 等旧模板没有这个变量，传了也不受影响。
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = requests.post(f"{self.base_url}/v1/chat/completions",
                             json=payload, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip(), data.get("usage") or {}