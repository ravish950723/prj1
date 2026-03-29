import py_compile, sys, pathlib

files = ['buy.py','symbol_analysis.py','buy_srs_update_excel.py']
root = pathlib.Path(__file__).resolve().parent
ok=True
for f in files:
    p=root/f
    try:
        py_compile.compile(str(p), doraise=True)
        print(f'[OK] {f}')
    except Exception as e:
        ok=False
        print(f'[FAIL] {f}: {e}')
if not ok:
    sys.exit(1)
