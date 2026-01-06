@echo off
chcp 65001 >nul
echo ========================================
echo 交互式聚类分析服务器
echo ========================================
echo.
echo 正在启动服务器...
echo.
echo 请确保已安装 Flask 和 flask-cors:
echo   pip install flask flask-cors
echo.
echo ========================================
echo.

cd /d "%~dp0"
python clustering_server.py

pause
