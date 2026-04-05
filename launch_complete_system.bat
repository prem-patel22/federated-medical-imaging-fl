@echo off
title Federated Medical Imaging System
echo ========================================
echo   🏥 FEDERATED MEDICAL IMAGING SYSTEM
echo ========================================
echo.
echo Choose an option:
echo.
echo [1] Start Training (Server + 3 Hospitals)
echo [2] Launch Dashboard
echo [3] Test & Save Model
echo [4] Run Grad-CAM Demo
echo [5] Run Privacy Demo
echo [6] Launch Everything!
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto training
if "%choice%"=="2" goto dashboard
if "%choice%"=="3" goto test
if "%choice%"=="4" goto gradcam
if "%choice%"=="5" goto privacy
if "%choice%"=="6" goto all

:training
echo Starting Federated Training...
start "Server" cmd /k "call fl_venv\Scripts\activate && python server\server.py"
timeout /t 2
start "Hospital A" cmd /k "call fl_venv\Scripts\activate && python clients\hospital_a.py"
start "Hospital B" cmd /k "call fl_venv\Scripts\activate && python clients\hospital_b.py"
start "Hospital C" cmd /k "call fl_venv\Scripts\activate && python clients\hospital_c.py"
goto end

:dashboard
echo Launching Dashboard...
call fl_venv\Scripts\activate
streamlit run dashboard\app.py
goto end

:test
echo Testing and Saving Model...
call fl_venv\Scripts\activate
python scripts\save_and_test_model.py
goto end

:gradcam
echo Running Grad-CAM Demo...
call fl_venv\Scripts\activate
python explainability\gradcam.py
goto end

:privacy
echo Running Privacy Demo...
call fl_venv\Scripts\activate
python privacy\differential_privacy.py
goto end

:all
echo Launching Complete System...
start "Server" cmd /k "call fl_venv\Scripts\activate && python server\server.py"
timeout /t 2
start "Hospital A" cmd /k "call fl_venv\Scripts\activate && python clients\hospital_a.py"
start "Hospital B" cmd /k "call fl_venv\Scripts\activate && python clients\hospital_b.py"
start "Hospital C" cmd /k "call fl_venv\Scripts\activate && python clients\hospital_c.py"
timeout /t 5
start "Dashboard" cmd /k "call fl_venv\Scripts\activate && streamlit run dashboard\app.py"
goto end

:end
echo.
echo ✅ Done!
pause