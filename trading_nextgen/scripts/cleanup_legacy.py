from __future__ import annotations

import argparse
from pathlib import Path

LEGACY_FILES = [
    'buy.py', 'buy_updated.py', 'buy_rl.py', 'buy_srs_update_excel.py', 'build_master_features.py',
    'validate_excel_columns.py', 'compile_check.py', 'run_all.bat', 'retrain_model.bat',
    'train_rl_policy.bat', 'testmodel.py', 'clean_pipeline.zip'
]
LEGACY_DIRS = ['clean_pipeline', '__pycache__']


def main() -> int:
    ap = argparse.ArgumentParser(description='Delete duplicated legacy files from an old monolithic workspace.')
    ap.add_argument('--root', required=True, help='Root folder of the old project to clean.')
    args = ap.parse_args()
    root = Path(args.root)
    removed = []
    for name in LEGACY_FILES:
        p = root / name
        if p.exists() and p.is_file():
            p.unlink()
            removed.append(str(p))
    for name in LEGACY_DIRS:
        p = root / name
        if p.exists() and p.is_dir():
            import shutil
            shutil.rmtree(p)
            removed.append(str(p))
    print('Removed:')
    for x in removed:
        print(x)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
