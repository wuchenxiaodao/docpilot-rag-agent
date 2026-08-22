\# Day 1 运行记录

\## 环境

\- 操作系统：Windows 11

\- 项目 Python：3.12.13

\- 系统默认 Python：3.9.13

\- uv：0.11.28

\- Git：2.54.0.windows.1

\- 项目分支：feature/docpilot

\## 环境隔离

论文项目继续使用原来的 Conda 环境。

DocPilot 使用 uv 管理的 Python 3.12.13、项目目录下的 `.venv` 和单独的 `.env`，没有向论文环境安装依赖。

\## 项目配置

\## 验证结果

\- 后端启动：成功

\- Swagger `/docs`：成功

\- ReDoc `/redoc`：成功

\- Streamlit 页面：成功

\- Fake Model 响应：成功

\- 后端端口：8080

\- Streamlit 端口：8501

\## 遇到的问题

\### 1. uv 命令无法识别

\- 现象：PowerShell 无法识别 `uv`。

\- 原因：电脑尚未安装 uv。

\- 解决：使用官方脚本安装 uv，然后重新打开 PowerShell。

\### 2. 担心影响论文环境

\- 原因：电脑中同时存在 Conda、Python 3.9 和 uv 管理的 Python 3.12。

\- 解决：为 DocPilot 创建独立 `.venv`，并明确使用 `.venv\\Scripts\\python.exe`。

\### 3. 访问后端根路径得到 404

\- 现象：访问 `http://localhost:8080/` 返回 404。

\- 原因：服务器已经启动，但项目没有定义 `/` 路由。

\- 解决：访问 `/docs` 或 `/redoc`。

\### 4. Streamlit 询问邮箱

\- 现象：首次启动停在 `Email:`。

\- 原因：这是 Streamlit 的可选欢迎信息。

\- 解决：不填写邮箱，直接按 Enter。

\## 我理解的虚拟环境

`.venv` 保存项目专属的 Python 运行入口、第三方依赖和命令行工具。

激活虚拟环境主要是调整当前终端的 PATH。即使不激活，也可以明确运行 `.venv\\Scripts\\python.exe`。

虚拟环境能隔离 Python 依赖，但不能隔离端口、CPU、内存或 GPU。

\## 我理解的调用链

浏览器 → Streamlit → HTTP Client → FastAPI → Agent → Fake Model → 返回结果
