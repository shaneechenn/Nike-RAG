@echo off
cd /d "%~dp0"
echo ============================================
echo   Nike Assistant - starting server...
echo   Keep this window open.
echo   Open http://localhost:8501 in your browser
echo   (Press Ctrl+C to stop the server)
echo ============================================
start "" "http://localhost:8501"
"C:\Users\shane\miniconda3\envs\nike-rag\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8501
pause
