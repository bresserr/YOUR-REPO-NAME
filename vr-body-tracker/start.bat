@echo off
echo Starting VR Body Tracker...
echo ================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create output directory if it doesn't exist
if not exist "output" mkdir output

REM Start the application
echo Starting server...
cd backend
python main.py
pause