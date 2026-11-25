@echo off
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed or not on PATH. & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements-gpu-windows.txt ctranslate2 faster-whisper
echo.
echo Checking the GPU...
python gpu_check.py
echo.
pause
