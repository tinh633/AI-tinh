@echo off
title AI Traffic Assistant - MCP + Flask Starter

echo ============================================
echo      BAT FILE - RUN MCP + FLASK SERVER
echo ============================================

:: ---- CHECK PYTHON ----
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Python không được cài hoặc chưa add PATH.
    pause
    exit /b
)

echo ✔ Python OK
echo.

:: ---- START MCP SERVER ----
echo Đang chạy MCP Server (luật)...
start cmd /k "python mcp_server.py"

:: ---- DELAY CHO MCP KHỞI ĐỘNG ----
timeout /t 2 >nul

:: ---- START FLASK SERVER ----
echo Đang chạy Flask Backend...
start cmd /k "python app.py"

echo.
echo ============================================
echo        ✨ CẢ HAI SERVER ĐÃ ĐƯỢC KHỞI ĐỘNG
echo.
echo  - MCP chạy tại:    http://127.0.0.1:8000
echo  - Flask chạy tại:  http://127.0.0.1:5000
echo ============================================
echo.

pause
