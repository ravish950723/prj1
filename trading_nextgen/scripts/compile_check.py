import py_compile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = sorted([p for p in root.rglob('*.py') if '.venv' not in str(p)])
failed = False
for p in files:
    try:
        py_compile.compile(str(p), doraise=True)
        print('[OK]', p.relative_to(root))
    except Exception as e:
        failed = True
        print('[FAIL]', p.relative_to(root), e)
if failed:
    raise SystemExit(1)
