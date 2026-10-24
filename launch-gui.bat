@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Run setup-iris.ps1 first to create the Python environment.
    pause
    exit /b 1
)
start "Visual Anagrams" ".venv\Scripts\pythonw.exe" "%~dp0gui.py"
