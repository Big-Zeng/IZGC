@echo off
chcp 65001 >nul
echo ========================================
echo 正在启动本地地图服务器...
echo ========================================
echo.

python start_server.py

pause
