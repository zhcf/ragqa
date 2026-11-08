@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

set "STOPPED=0"

if exist ".app.pid" (
    set /p APP_PID=<".app.pid"
    echo 正在停止进程树 !APP_PID! ...
    taskkill /F /T /PID !APP_PID! >nul 2>&1
    tasklist /FI "PID eq !APP_PID!" 2>nul | find "!APP_PID!" >nul
    if errorlevel 1 (
        set "STOPPED=1"
        del ".app.pid" >nul 2>&1
    ) else (
        echo 无法结束进程 !APP_PID!，请右键以管理员身份运行本脚本。
    )
)

rem 主进程异常退出时 llama-server 可能会残留，一并清掉
tasklist /FI "IMAGENAME eq llama-server.exe" 2>nul | find /I "llama-server.exe" >nul
if not errorlevel 1 (
    echo 清理残留的 llama-server ...
    taskkill /F /IM llama-server.exe >nul 2>&1
    set "STOPPED=1"
)

if "!STOPPED!"=="1" (
    echo 已停止。
) else (
    echo 没有发现正在运行的实例。
)
exit /b 0