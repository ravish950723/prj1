import sys
import pandas as pd
import numpy as np

def pick_sheet(path: str, preferred: str = "Summary") -> str:
    """Pick the best sheet name to validate.
    - If preferred exists, use it.
    - Else prefer 'Sheet1' if exists (common in this repo).
    - Else pick the first sheet.
    """
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        names = wb.sheetnames
        if preferred in names:
            return preferred
        if "Sheet1" in names:
            return "Sheet1"
        return names[0] if names else preferred
    except Exception:
        # fallback: pandas will raise a helpful error if sheet is wrong
        return preferred

def main():
    path = "predictions_summary_out.xlsx"
    preferred_sheet = "Summary"

    # CLI: validate_excel_columns.py <path> [--sheet SHEETNAME]
    args = sys.argv[1:]
    if args:
        # first positional if not flag
        if not args[0].startswith("--"):
            path = args[0]
            args = args[1:]

    if "--sheet" in args:
        i = args.index("--sheet")
        if i + 1 < len(args):
            preferred_sheet = args[i + 1]

    sheet = pick_sheet(path, preferred=preferred_sheet)
    print(f"[INFO] Validating file: {path}")
    print(f"[INFO] Using sheet: {sheet}")

    df = pd.read_excel(path, sheet_name=sheet)

    # Required columns (keep your existing list)
    req = [
        'DMA20', 'DMA50', 'DMA200', 'PCT_FROM_DMA50', 'PCT_FROM_DMA200',
        'VOL_TODAY', 'AVG_VOL_20D', 'VOL_SURGE_RATIO',
        'VWAP', 'VWAP_DISTANCE_PCT',
        'ATR14_PCT',
        'DISTRIBUTION_DAYS_20D', 'ACCUMULATION_DAYS_20D',
        'LONG_SCORE', 'SHORT_SCORE',
        'FINAL_ACTION', 'CONFIDENCE_SCORE'
    ]

    # Normalize existing columns for robustness
    cols = set(df.columns.astype(str))
    missing_cols = [c for c in req if c not in cols]

    print(f"[INFO] Rows: {len(df)}  Cols: {len(df.columns)}")
    print(f"[INFO] Missing cols: {missing_cols}")

    # Blank/NA report for present columns
    present = [c for c in req if c in cols]
    if not present:
        print("[WARN] None of the required columns were found on this sheet.")
        sys.exit(2)

    report = []
    for c in present:
        s = df[c]
        # Treat '', 'N/A', 'NA', None as blank-ish
        blankish = s.isna() | (s.astype(str).str.strip().isin(["", "N/A", "NA", "NONE", "nan"]))
        report.append((c, int(blankish.sum()), int(len(s) - blankish.sum())))

    # Print worst offenders first
    report.sort(key=lambda x: x[1], reverse=True)

    print("\n[BLANK REPORT] (column | blankish | filled)")
    for c, b, f in report:
        print(f"{c:28s} | {b:5d} | {f:5d}")

    # Non-zero exit if missing required columns
    if missing_cols:
        sys.exit(1)

if __name__ == "__main__":
    main()
