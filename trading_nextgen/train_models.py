from __future__ import annotations

import argparse
import json
from pathlib import Path

from buypipe.train_ml import train_ml_model
from buypipe.train_dl import train_dl_model
from buypipe.train_rl import train_rl_policy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='Excel or CSV source with historical labeled rows')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--skip-ml', action='store_true')
    ap.add_argument('--skip-dl', action='store_true')
    ap.add_argument('--skip-rl', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    if not args.skip_ml:
        summary['ml'] = train_ml_model(args.source, str(out_dir))
    if not args.skip_dl:
        summary['dl'] = train_dl_model(args.source, str(out_dir), epochs=args.epochs)
    if not args.skip_rl:
        summary['rl'] = train_rl_policy(args.source, str(out_dir))

    with open(out_dir / 'training_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
