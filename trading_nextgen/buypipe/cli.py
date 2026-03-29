from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import main as pipeline_main
from .train_ml import train_ml_model
from .train_dl import train_dl_model


def run_pipeline_cli() -> int:
    return pipeline_main()


def train_ml_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="train_data.csv")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    metrics = train_ml_model(args.source, args.out_dir)
    print(metrics)
    return 0


def train_dl_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="train_data.csv")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    metrics = train_dl_model(args.source, args.out_dir, epochs=args.epochs)
    print(metrics)
    return 0


def train_models_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="train_data.csv")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    train_ml_model(args.source, args.out_dir)
    return 0