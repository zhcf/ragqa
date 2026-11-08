@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

if exist ".app.pid" (
    set /p APP_PID=<".app.pid"
    tasklist /FI "PID eq !APP_PID!" 2>nul | find "!APP_PID!" >nul
    if not errorlevel 1 (
        echo 程序已在运行（PID !APP_PID!），如需重启请先运行 stop.bat
        pause
        exit /b 1
    )
    rem 进程已不在，清掉上次残留的记录
    del ".app.pid" >nul 2>&1
)

if not exist "env\Scripts\pythonw.exe" (
    echo 找不到虚拟环境：env\Scripts\pythonw.exe
    pause
    exit /b 1
)

echo 正在启动 PDF 手册问答...
start "" "%~dp0env\Scripts\pythonw.exe" "%~dp0desktop.py"
exit /b 0