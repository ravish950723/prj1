@echo off
cd /d "%~dp0"

echo [1/3] Generating training data...
python -m buypipe.generate_training_data --out train_data.csv
if errorlevel 1 goto :fail

echo [2/3] Training ML + DL models...
buypipe-train --source train_data.csv --out-dir . --epochs 40
if errorlevel 1 goto :fail

echo [3/3] Done.
goto :eof

:fail
echo Training failed.
exit /b 1