# ML Breakouts

Machine-learning meta-model that filters horizontal-resistance breakout setups
on Indian equities (NSE main board). Built to be evaluated honestly (purged
walk-forward CV with embargo) and deployed daily.

## Architecture

```
            ┌────────────────────────────────────────────┐
            │ PRIMARY  (rule-based, deterministic)       │
            │ - resistance detection (fractal pivots)    │
            │ - base-formation gate                      │
            │ → emits "this is a setup" Y/N              │
            └────────────────┬───────────────────────────┘
                             │ only setups continue
                             ▼
            ┌────────────────────────────────────────────┐
            │ META-MODEL (LightGBM, calibrated)          │
            │ - 50+ engineered features                  │
            │ - triple-barrier classification labels     │
            │ - purged k-fold CV with embargo            │
            │ → P(profitable breakout)                   │
            └────────────────┬───────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        P > THRESHOLD → TRADE     P ≤ THRESHOLD → SKIP
        size = f(P, ATR)          (filters out chop)
```

## Why LightGBM (not Random Forest, not Transformers)

| Model            | Verdict |
|------------------|---------|
| Random Forest    | Outdated since 2017. Loses to GBM on every tabular benchmark. |
| **LightGBM**     | Industry standard for daily equity alpha. Used by WorldQuant, Two Sigma, Numerai, Kaggle JaneStreet/Optiver/JPX winners. |
| XGBoost          | Equivalent to LightGBM, slightly slower. |
| CatBoost         | Strong alternative when categorical features dominate. |
| TabNet           | Hype model. Loses to LGBM in Shwartz-Ziv & Armon (NeurIPS 2022). |
| TFT / Transformer| Needs ≥50k samples and GPUs. We have ~5k setups. Massive overkill. |
| LSTM / GRU       | Replaced by TCN/Transformer; both lose to LGBM on tabular data. |

LightGBM as the **meta-model**, not as a price predictor. We model
`P(setup succeeds)` not `next-day return`.

## Project Layout

```
ML Breakouts/
├── configs/
│   └── default.yaml          # all knobs in one place
├── data/
│   ├── ohlcv/                # cached daily bars (parquet, .gitignored)
│   ├── training/             # built (features, label) parquet
│   └── models/               # serialized LGBM + isotonic calibrator
├── src/
│   ├── data_ingestion.py     # NSE bhavcopy fetcher (no per-symbol API)
│   ├── universe.py           # NSE-EQ universe builder + filters
│   ├── setup_detector.py     # rule-based primary (port from Analysis/)
│   ├── features.py           # feature engineering
│   ├── labeling.py           # triple-barrier + sample uniqueness
│   ├── cv.py                 # purged k-fold + embargo
│   ├── model.py              # LGBM trainer + isotonic calibration
│   ├── backtest.py           # walk-forward dataset builder
│   └── inference.py          # daily scan
├── scripts/
│   ├── 01_build_dataset.py   # historical sweep → training parquet
│   ├── 02_train_model.py     # train LGBM with purged CV
│   ├── 03_evaluate.py        # OOS metrics + calibration plots
│   └── 04_daily_scan.py      # production scan
├── tests/
└── requirements.txt
```

## Phased plan

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1     | Data ingestion (NSE bhavcopy, no WAF block) + universe builder | 🔨 In progress |
| 2     | Setup detector ported & feature engineering (50+ features) | TODO |
| 3     | Triple-barrier labeling + purged CV harness | TODO |
| 4     | Walk-forward dataset builder → training parquet | TODO |
| 5     | LightGBM baseline with isotonic calibration | TODO |
| 6     | Evaluation: precision/recall, IC, R-multiple, calibration plots | TODO |
| 7     | Daily inference script | TODO |
| 8     | (optional) DoubleEnsemble / CatBoost comparison | TODO |
| 9     | (optional) Position sizing via half-Kelly | TODO |

## References (read these, not Medium articles)

- López de Prado, *Advances in Financial Machine Learning* (2018) — Ch. 3 (triple-barrier), Ch. 4 (sample weighting), Ch. 7 (purged CV)
- Microsoft Qlib model zoo — https://github.com/microsoft/qlib
- Shwartz-Ziv & Armon, *Tabular Data: Deep Learning is Not All You Need* (NeurIPS 2022)
- Numerai Whitepaper — feature neutralization & meta-modeling
- Bensdorp, *Automated Stock Trading Systems* — for trade-bookkeeping discipline


## Operational cadence — when to run what

![Operational cadence table](docs/images/operational_cadence.png)

**Quick combos:**
- **Daily routine**: `05_daily_scan.py` then `08_sector_rotation.py` -> produces `Output/triggers_*.csv` + `Output/watchlist_in_leaders_*.csv`
- **Quarterly refresh** (full): `06 -> 01 -> 02 -> 03 -> 04`
- **Yearly tune**: `07 -> 03 -> 04`


