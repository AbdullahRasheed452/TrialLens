@echo off
rem TrialLens launcher: sets up the environment on first run, then opens the app in the browser.
setlocal
cd /d "%~dp0"
title TrialLens

if not exist ".venv\Scripts\python.exe" (
    echo [TrialLens] First run: creating the Python environment. This takes a few minutes.
    where uv >nul 2>nul
    if %errorlevel%==0 (
        uv venv --python 3.11 .venv || goto :fail
        uv pip install --python .venv -r requirements.txt || goto :fail
    ) else (
        python -m venv .venv || goto :fail
        ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
    )
)

if not exist "data\processed\val.jsonl" (
    echo [TrialLens] Building the dataset split.
    ".venv\Scripts\python.exe" scripts\make_split.py || goto :fail
)

if not exist "checkpoints\BEST" (
    echo [TrialLens] Downloading the fine-tuned adapter.
    ".venv\Scripts\python.exe" scripts\download_adapter.py
)

echo [TrialLens] Starting. The browser opens at http://127.0.0.1:7860 once the model is loaded.
".venv\Scripts\python.exe" app\app.py %*
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo [TrialLens] Something went wrong. See the messages above, or the Troubleshooting section of README.md.
pause
exit /b 1
