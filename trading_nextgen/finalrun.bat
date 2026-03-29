@echo off
cd /d "%~dp0"

echo =========================================
echo   BUYPIPE NEXTGEN FULL PIPELINE START
echo =========================================

REM -------------------------------
REM Step 1: Generate Training Data
REM -------------------------------
echo [1/4] Generating training data...
python -m buypipe.generate_training_data --out train_data.csv
if errorlevel 1 goto :fail

REM -------------------------------
REM Step 2: Train Models
REM -------------------------------
echo [2/4] Training ML + DL models...
buypipe-train --source train_data.csv --out-dir . --epochs 40
if errorlevel 1 goto :fail

REM -------------------------------
REM Step 3: Activate Virtual Env
REM -------------------------------
echo [3/4] Activating virtual environment...
call .venv\Scripts\activate
if errorlevel 1 goto :fail

REM -------------------------------
REM Step 4: Run Prediction Pipeline
REM -------------------------------
echo [4/4] Running prediction pipeline...
buypipe-run ^
  --template predictions_summary_out.xlsx ^
  --schema configs/col.yml ^
  --out outputs/predictions_summary_out_nextgen.xlsx
if errorlevel 1 goto :fail

echo =========================================
echo   PIPELINE COMPLETED SUCCESSFULLY
echo =========================================
goto :eof

:fail
echo =========================================
echo   PIPELINE FAILED
echo =========================================
exit /b 1