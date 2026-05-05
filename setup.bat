@echo off
REM =============================================================
REM  setup.bat — AI Meme Emote Detector (Windows)
REM  Run from inside the emote-meme\ folder:
REM    cd emote-meme
REM    setup.bat
REM =============================================================

echo.
echo ============================================================
echo   AI Meme Emote Detector — Project Setup
echo ============================================================
echo.

REM --- 1. Create folder structure ---
echo [1/4] Creating project folders...
if not exist images  mkdir images
if not exist models  mkdir models
echo       images\  and  models\  are ready.

REM --- 2. Create virtual environment ---
echo.
echo [2/4] Creating virtual environment (venv)...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo        Download Python 3.11+ from https://python.org
    pause
    exit /b 1
)
echo       venv\ created.

REM --- 3. Install dependencies ---
echo.
echo [3/4] Installing dependencies from requirements.txt...
call venv\Scripts\activate.bat
pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Check your internet connection.
    pause
    exit /b 1
)
echo       Dependencies installed.

REM --- 4. Done ---
echo.
echo [4/4] Setup complete!
echo.
echo ============================================================
echo   HOW TO RUN
echo ============================================================
echo.
echo   1. Activate virtual environment:
echo        venv\Scripts\activate
echo.
echo   2. Drop meme images into the  images\  folder
echo      (see GESTURE_MEME_MAP in main.py for expected filenames)
echo.
echo   3. Run the app:
echo        python main.py
echo.
echo   4. Press 'q' inside the window to quit.
echo.
echo   NOTE: MediaPipe model files are auto-downloaded on first run.
echo ============================================================
echo.
pause
