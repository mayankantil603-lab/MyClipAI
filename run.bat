@echo off
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed or not on PATH. Install it from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" (
  echo First run: setting up, this takes a few minutes...
  python -m venv .venv || (pause & exit /b 1)
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt || (pause & exit /b 1)
) else (
  call .venv\Scripts\activate.bat
  python -c "import PySide6, faster_whisper, yt_dlp, keyring" >nul 2>nul || (
    echo Repairing the setup...
    python -m pip install -r requirements.txt || (pause & exit /b 1)
  )
)
where nvidia-smi >nul 2>nul && (
  python -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('nvidia.cublas') and u.find_spec('nvidia.cudnn') else 1)" >nul 2>nul || (
    echo NVIDIA GPU found. Installing GPU support, one time, about 1 GB...
    python -m pip install -r requirements-gpu-windows.txt
  )
)
if not exist ".venv\shortcuts_v1" (
  echo Adding MyClip AI to your Desktop and Start menu...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_shortcuts.ps1" -Quiet && echo done> ".venv\shortcuts_v1"
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0MyClipAI.pyw"
