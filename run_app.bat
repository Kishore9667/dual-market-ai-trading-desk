@echo off
title Institutional Dual-Market AI Trading Desk (US & India)
set OLLAMA_FLASH_ATTENTION=1
cd /d "%~dp0"
echo Starting Institutional Dual-Market AI Trading Desk...
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
pause
