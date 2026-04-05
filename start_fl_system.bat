@echo off
echo ========================================
echo   FEDERATED MEDICAL IMAGING SYSTEM
echo ========================================
echo.
echo Starting Federated Learning System...
echo.

REM Activate virtual environment
call fl_venv\Scripts\activate

REM Run the system
python scripts\run_all.py

pause