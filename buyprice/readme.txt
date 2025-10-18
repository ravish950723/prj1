README — AI Trading Signal Pipeline
====================================

This repository contains a complete end‑to‑end pipeline to:
 • Fetch historical OHLCV data from Interactive Brokers (IBKR)
 • Compute 40+ technical features and signals
 • Generate supervised labels and training data
 • Train and calibrate an XGBoost “Strong Buy” classifier
 • Run live inference with rule-based + ML signals
 • Backtest short‑term hit rates and export a ranked summary CSV

Files & Roles (Quick Map)
-------------------------
• config.py                    – global config, symbol lists, sector maps, thresholds
• fetching.py                  – IBKR data ingestion + local CSV caching
• compute.py                   – indicator & signal engine (+ market stage, candle entries)
• darvas.py                    – Darvas box breakout features
• institutional_investor.py    – institutional accumulation score
• upward.py                    – SMC/mean-reversion/candlestick signal helpers
• generate_training_data.py    – build train_data.csv (+ SMOTE option)
• train_strong_buy_model.py    – train & calibrate model; export thresholds JSON + model .pkl
• testmodel.py                 – tiny model sanity test with named features
• symbol_analysis.py           – single-symbol analytics (buy price, correlation, backtest)
• backtest.py                  – simple forward 90D hit/gain metrics helper
• buy.py                       – main pipeline runner; prints table & saves predictions_summary.csv


1) Prerequisites
----------------
A) Software
   • Python 3.10+ recommended
   • IBKR TWS or IB Gateway (paper/live). Enable API access:
     TWS → File → Global Configuration → API → Settings
       [✓] Enable ActiveX and Socket Clients
       Socket port: 7497 (paper) or as you prefer
       Read‑Only API is OK
   • Internet access for TWS/Gateway to download market data (delayed ok)

B) Python packages
   Create & activate a virtual environment, then install deps:

   Windows (PowerShell):
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install --upgrade pip
     pip install pandas numpy pandas_ta ib_insync xgboost scikit-learn imbalanced-learn joblib matplotlib

   macOS/Linux:
     python3 -m venv .venv
     source .venv/bin/activate
     pip install --upgrade pip
     pip install pandas numpy pandas_ta ib_insync xgboost scikit-learn imbalanced-learn joblib matplotlib

   Notes:
     • TA-Lib is NOT required; the project uses pandas_ta.
     • If you hit compiler errors for xgboost on macOS, install via
       pip install xgboost==2.0.* (or brew install libomp)


2) Configure Environment
------------------------
Optional environment variables (override defaults from config.py):
 • CACHE_DIR   – default "cache"
 • IB_HOST     – default "127.0.0.1"
 • IB_PORT     – default 7497 (paper); 7496 for live
 • IB_CLIENT_ID – default 103

Examples:

Windows (PowerShell):
  setx CACHE_DIR "cache"
  setx IB_HOST "127.0.0.1"
  setx IB_PORT "7497"
  setx IB_CLIENT_ID "103"

macOS/Linux (bash):
  export CACHE_DIR="cache"
  export IB_HOST="127.0.0.1"
  export IB_PORT="7497"
  export IB_CLIENT_ID="103"

Symbols & Sectors:
 • Edit config.py → `symbols` list, `symbol_to_sector`, and `sector_etfs` as needed.


3) Typical Run Sequences
------------------------

A) One‑time (or periodic) training
----------------------------------
1. Generate training data (chronological):
   python generate_training_data.py
   → Outputs:
     • train_data.csv (base, time‑ordered)
     • (optional) train_data_balanced.csv (SMOTE)
     • model_features.txt (feature names list)

2. Train & calibrate model:
   python train_strong_buy_model.py
   → Outputs:
     • strong_buy_xgb_model_calibrated.pkl
     • strong_buy_thresholds.json (best F1/Recall thresholds, e.g., 0.07 / 0.05)
   (Keeps chronological split to avoid leakage; uses isotonic calibration.)

3. (Optional) Quick smoke test:
   python testmodel.py
   → Writes strong_buy_xgb_model.pkl for name‑feature sanity checks (not required for live).


B) Daily (or on‑demand) inference
---------------------------------
1. Ensure IBKR TWS/Gateway is running and API is enabled.
2. Run the main analyzer:
   python buy.py
   → Actions:
     • Loads model + thresholds (if present)
     • For each symbol:
         - fetches 3Y/1D (uses cache if available)
         - computes indicators & signals
         - computes candle entries (6w/8w/12w/18w/30w)
         - runs ML prediction and tech‑fallback logic
         - evaluates a forward 90D backtest on the suggested buy price
     • Prints a ranked table
     • Saves CSV → predictions_summary.csv in the repo root

3. Inspect outputs:
   • predictions_summary.csv  – consolidated dashboard
   • cache/                   – one CSV per (symbol, duration, bar_size)


4) What Each Output Column Means (buy.py)
-----------------------------------------
Key columns (if present):
 • Symbol                 – ticker
 • Refined Buy Price      – blended anchor (EMA21, VWAP, Darvas, BB, swing‑low, candle entries)
 • Candle Entry 6w..30w   – anchor entries per window (conservative for 6/8w unless reversal)
 • VWAP Support           – 5‑bar swing‑low proxy
 • ADX                    – trend strength (≥25 strong)
 • Institutional Score    – OBV/volume/momentum blend (0–1)
 • Volume Weight          – volume / 20D avg (capped)
 • Confidence Score       – composite (institutional, volume, ADX)
 • Sector Correlation     – 20D corr to sector ETF
 • Trend / Market Stage   – UP/DOWN and (Accumulation/Mark‑Up/Distribution/Mark‑Down)
 • Darvas Breakout %/Signal – classic breakout diagnostics
 • Model Probability      – calibrated proba of “strong buy”
 • Confidence Band        – WATCH / HOLD / BUY / STRONG BUY
 • Rule‑Based Buy         – confirms multi‑trend + confidence thresholds
 • Model‑Driven Buy       – thresholded by STRONG_BUY
 • 90D Hit/Gain/Days      – forward window performance vs buy price


5) Troubleshooting
------------------
IBKR connection
 • Error: “clientId already in use” → change IB_CLIENT_ID in env or config.py
 • No bars returned / contract not qualified → symbol/exchange mismatch;
   fetching.py tries SMART/ARCA/ISLAND. Ensure market data permissions (delayed OK).

Model/Features
 • “Model file not found.” → run train_strong_buy_model.py or place the .pkl file in repo root.
 • Feature mismatch → ensure model_features.txt comes from the *same* training used to save the model.
   buy.py aligns columns using model.feature_names_in_ (preferred) or model_features.txt as fallback.

Packages
 • pandas_ta import error → pip install pandas_ta
 • xgboost build issues (macOS) → install libomp; try a slightly older xgboost (2.0.*).

Caching
 • To force fresh data for a symbol, delete its CSV in the cache/ folder or change `refresh=True` in calls.
 • To reset the entire cache, delete the cache/ directory.


6) Customization Knobs
----------------------
• Symbols/Sectors: edit config.py → `symbols`, `symbol_to_sector`, `sector_etfs`
• Buy‑band logic: see symbol_analysis.py (EMA21/VWAP/Darvas/BB/swing‑low + Fibonacci bias)
• Candle entries: compute.py → `candle_entry_from_weeks()` and `candle_entries_multi()`
• Darvas strictness: darvas.py (increase vol confirmation to 1.2× avg)
• Signal score weights: compute.py & upward.py
• Thresholds: strong_buy_thresholds.json (generated by training)
• IBKR router/exchange attempts: fetching.py → `_EXCHANGE_TRIES`


7) Example One‑Liners
---------------------
• Generate → Train → Predict (fresh venv):
  python -m venv .venv && . .venv/bin/activate && pip install -U pip && \
  pip install pandas numpy pandas_ta ib_insync xgboost scikit-learn imbalanced-learn joblib matplotlib && \
  python generate_training_data.py && \
  python train_strong_buy_model.py && \
  python buy.py

• Windows (PowerShell):
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install --upgrade pip
  pip install pandas numpy pandas_ta ib_insync xgboost scikit-learn imbalanced-learn joblib matplotlib
  python generate_training_data.py
  python train_strong_buy_model.py
  python buy.py


8) Suggested Daily Workflow
---------------------------
1. Update `symbols` in config.py (if needed)
2. Start IBKR TWS (paper), check API port
3. Run: python buy.py
4. Open predictions_summary.csv → sort by “Model Probability”, “Confidence Band”, “90D Gain (%)”
5. Validate entries vs chart; use candle entry prices as staggered limit orders


Appendix: File Integrity & Notes
--------------------------------
• Make sure `fetching.py` uses IB_HOST/IB_PORT from config/env (not hard‑coded).
• Avoid importing `model` from symbol_analysis within buy.py; use the local loader.
• Ensure train/test splits are chronological (already done in train_strong_buy_model.py).
• Keep feature definitions synchronized between training and inference.
• Use paper trading first; nothing here places orders automatically.

— End of README —
