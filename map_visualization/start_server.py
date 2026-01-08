"""
启动本地HTTP服务器来查看地图
解决file://协议的CORS跨域问题
"""

import http.server
import socketserver
import webbrowser
import os

# 配置
PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # 添加CORS头
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_server():
    """启动HTTP服务器"""
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print("="*60)
            print("本地HTTP服务器已启动")
            print("="*60)
            print(f"服务器地址: http://localhost:{PORT}")
            print(f"工作目录: {DIRECTORY}")
            print("\n可用的地图文件:")
            print(f"  - 百度地图: http://localhost:{PORT}/client_map_baidu.html")
            print(f"  - Leaflet地图: http://localhost:{PORT}/client_map_leaflet.html")
            print("\n按 Ctrl+C 停止服务器")
            print("="*60)
            
            # 自动打开浏览器
            url = f"http://localhost:{PORT}/client_map_baidu.html"
            print(f"\n正在打开浏览器: {url}")
            webbrowser.open(url)
            
            # 启动服务器
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
    except OSError as e:
        if e.errno == 10048:  # 端口已被占用
            print(f"\n错误: 端口 {PORT} 已被占用")
            print(f"请尝试:")
            print(f"  1. 关闭占用端口的程序")
            print(f"  2. 或修改脚本中的PORT变量为其他端口")
        else:
            print(f"\n错误: {e}")

if __name__ == "__main__":
    start_server()
