@echo off
setlocal

REM ==========================
REM Fully automated pipeline
REM No pauses, no interaction
REM ==========================

cd /d "%~dp0"

REM Pick Python: prefer venv if present
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

REM Logging
if not exist "logs" mkdir logs
set LOG=logs\run_all.log

echo =============================================== > "%LOG%"
echo RUN_ALL started: %DATE% %TIME% >> "%LOG%"
echo Working dir: %CD% >> "%LOG%"
echo Python: %PY% >> "%LOG%"
echo =============================================== >> "%LOG%"

REM ------------------------------------------------
REM [1/5] Refresh cache (delete cache folder)
REM ------------------------------------------------
echo [1/5] Refreshing cache...
echo [1/5] Refreshing cache... >> "%LOG%"

if exist "cache" (
    rmdir /s /q "cache" >> "%LOG%" 2>&1
)
mkdir "cache" >> "%LOG%" 2>&1

REM ------------------------------------------------
REM [2/5] Generate training data
REM ------------------------------------------------
echo [2/5] Generating training data...
echo [2/5] Generating training data... >> "%LOG%"

%PY% generate_training_data.py >> "%LOG%" 2>&1
if errorlevel 1 goto FAIL

REM ------------------------------------------------
REM [3/5] Train model
REM ------------------------------------------------
echo [3/5] Training model...
echo [3/5] Training model... >> "%LOG%"

%PY% train_strong_buy_model.py >> "%LOG%" 2>&1
if errorlevel 1 goto FAIL

REM ------------------------------------------------
REM [4/5] Test model
REM ------------------------------------------------
echo [4/5] Testing model...
echo [4/5] Testing model... >> "%LOG%"

%PY% testmodel.py >> "%LOG%" 2>&1
if errorlevel 1 goto FAIL

REM ------------------------------------------------
REM [5/5] Run buy pipeline
REM ------------------------------------------------
echo [5/5] Running buy.py...
echo [5/5] Running buy.py... >> "%LOG%"

%PY% buy.py >> "%LOG%" 2>&1
if errorlevel 1 goto FAIL

echo =============================================== >> "%LOG%"
echo ✅ ALL DONE: %DATE% %TIME% >> "%LOG%"
echo Output: predictions_summary_out.xlsx >> "%LOG%"
echo =============================================== >> "%LOG%"

exit /b 0

REM ------------------------------------------------
REM FAILURE HANDLER
REM ------------------------------------------------
:FAIL
echo =============================================== >> "%LOG%"
echo ❌ PIPELINE FAILED at %DATE% %TIME% >> "%LOG%"
echo Check log: %LOG% >> "%LOG%"
echo =============================================== >> "%LOG%"
exit /b 1
