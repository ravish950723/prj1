@echo off
call .venv\Scripts\activate
buypipe-run --template predictions_summary_out.xlsx --schema configs/col.yml --out outputs/predictions_summary_out_nextgen.xlsx