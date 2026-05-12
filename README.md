# ML Breakouts

End-to-end production system for ranking horizontal-resistance breakout
setups on Indian equities (NSE main board). Combines a rule-based primary
detector, a calibrated LightGBM meta-model, sector + theme rotation
context, and a parallel pre-breakout scanner. Runs daily via launchd.

---

## 1. High-level architecture

```
                    ┌──────────────────────────────────────┐
                    │  DATA LAYER  (data/)                 │
                    │  • OHLCV cache  (Angel One)          │
                    │  • Benchmark ^NSEI                   │
                    │  • Fundamentals + sector mapping     │
                    │  • Sector / theme equal-weight idx   │
                    └─────────────────┬────────────────────┘
                                      │
            ┌─────────────────────────┴──────────────────────────┐
            ▼                                                    ▼
   ┌─────────────────────┐                          ┌──────────────────────┐
   │  ML PIPELINE        │                          │  LEGACY SCANNER      │
   │  scripts/01..07     │                          │  legacy_scanner/     │
   │                     │                          │                      │
   │  Setup detector →   │                          │  Pre-breakout v3.5:  │
   │  ~110 features →    │                          │  hard gates →        │
   │  LightGBM +         │                          │  pattern detectors → │
   │  isotonic +         │                          │  HC / WL classifier  │
   │  stacker + regime   │                          │                      │
   │  + R-multiple head  │                          └──────────────────────┘
   └────────┬────────────┘
            │
            ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  DAILY SCAN  (scripts/05_daily_scan.py)                     │
   │  for each symbol:                                            │
   │    OHLCV tail-update → setup → features → predict P(win)    │
   │  filter @ watchlist / trigger thresholds → risk plan        │
   │  CHAIN → 08_sector_rotation.py                              │
   │             ├─ GICS sector rotation                         │
   │             ├─ Custom theme rotation                        │
   │             └─ MLRotation_<date>.xlsx (combined workbook)   │
   └─────────────────────────────────────────────────────────────┘
            │
            ▼
   Output/  scan_*.csv, triggers_*.csv,
            sector_rotation_*.csv, theme_rotation_*.csv,
            watchlist_in_leaders_*.csv,
            watchlist_in_theme_leaders_*.csv,
            MLRotation_<date>_<time>.xlsx
```

The two systems (ML pipeline + legacy_scanner) are independent. The ML
pipeline is the canonical daily output; the legacy scanner is a
complementary, rule-only pre-breakout audit that does not depend on a
trained model.

---

## 2. Data sources

| Source | Used for | Auth |
|---|---|---|
| **Angel One SmartAPI** (primary) | Daily OHLCV, ^NSEI benchmark; ~2 req/sec | `.env`: `ANGEL_API_KEY`, `ANGEL_CLIENT_CODE`, `ANGEL_PIN`, `ANGEL_TOTP_SECRET` (TOTP via `pyotp`) |
| **jugaad-data** (fallback 1) | NSE main board only | None |
| **yfinance** (fallback 2) | Broad coverage; fundamentals & market-cap (Angel does NOT expose these) | None |
| **NSE archives CSV** | Universe seed (NSE main + NSE Emerge SME) | None |
| **BSE listing JSON** | BSE SME platform + NSE→BSE fallback map | None |
| **NSE F&O list CSV** | Drop F&O underlyings from pct-down screener | None |
| **^CRSLDX (Yahoo)** | NIFTY 500 RS baseline (pct-down screener) | None |
| **NSE price-band-hitter API** | Upper-circuit deep-dive (`upper_band/`) | None |

OHLCV routing is centralized in `legacy_scanner/data_provider.py`
(`download(...)` → Angel → jugaad → yfinance). Mirror module
`src/yf_ingestion.py` is used by the ML pipeline.

---

## 3. Repository layout

```
ML Breakouts/
├── configs/
│   └── default.yaml              # ALL ML knobs: universe, setup gates,
│                                 # labeling, CV, model params, inference
│                                 # thresholds, Kelly sizing
│
├── data/                         # local cache + ML artefacts
│   ├── ohlcv/<SYM>.parquet       # ~2500 daily-bar parquets (Angel One)
│   ├── benchmark_NIFTY.parquet   # ^NSEI close
│   ├── fundamentals.parquet      # yfinance .info snapshot + ranks
│   ├── sector_indices.parquet    # equal-weight GICS sector levels
│   ├── theme_indices.parquet     # equal-weight custom themes
│   ├── sectors_seed.json         # legacy sector seed (deprecated)
│   ├── training/
│   │   └── setups.parquet        # ~24k rows × 119 cols (replay output)
│   └── models/                   # trained artefacts (12 files)
│       ├── lgbm.txt              # final LightGBM
│       ├── lgbm_regime_bull.txt  # regime sub-models
│       ├── lgbm_regime_bear.txt
│       ├── calibrator.pkl        # isotonic
│       ├── regime_calibrators.pkl
│       ├── stacker.pkl           # LR meta-learner over LGBM/XGB
│       ├── rmult_regressor.txt   # R-multiple head (Kelly sizing)
│       ├── features.json         # feature column list
│       ├── best_hparams.yaml     # latest Optuna winner
│       ├── oof_predictions.parquet  # CV OOF predictions for evaluation
│       ├── cv_metrics.csv
│       └── feature_importance.csv
│
├── src/                          # ML pipeline modules
│   ├── paths.py                  # config + canonical paths
│   ├── data_ingestion.py         # universe derivation from cache
│   ├── yf_ingestion.py           # OHLCV download / tail-update
│   ├── universe.py               # universe filters
│   ├── setup_detector.py         # rule-based primary (resistance + base)
│   ├── features.py               # ~110 feature engineering pipeline
│   ├── labeling.py               # triple-barrier + sample uniqueness
│   ├── cv.py                     # walk-forward + purged k-fold w/ embargo
│   ├── model.py                  # LightGBM trainer + isotonic + stacker
│   │                             # + regime sub-models + R-multiple head
│   ├── fundamentals.py           # yfinance .info fetcher + ranks
│   ├── sectors.py                # GICS sector index builder
│   ├── sector_rotation.py        # full sector rotation pipeline
│   ├── theme_rotation.py         # custom theme rotation pipeline
│   └── index_constituents.json   # custom theme baskets
│
├── scripts/                      # phase-numbered driver scripts
│   ├── 01_build_dataset.py       # bootstrap/refresh OHLCV cache
│   ├── 02_build_training_set.py  # replay history → setups.parquet
│   ├── 03_train_model.py         # train LGBM + calibrator + heads
│   ├── 04_evaluate.py            # OOF metrics, decile, calibration
│   ├── 05_daily_scan.py          # PRODUCTION daily scan + chains 08
│   ├── 06_fetch_fundamentals.py  # refresh fundamentals + sector idx
│   ├── 07_optuna_search.py       # LGBM hyperparam search
│   ├── 08_sector_rotation.py     # sector + theme rotation report
│   └── run_daily_scan.sh         # launchd entry point
│
├── legacy_scanner/               # rule-only pre-breakout system
│   ├── angel_client.py           # SmartAPI wrapper (TOTP, scrip-master)
│   ├── data_provider.py          # Angel → jugaad → yfinance fallback
│   ├── breakout_scanner_angel.py # v3.5 HC/WL classifier (live scanner)
│   ├── multi_pct_down.py         # pct-down screener (NSE/SME/BSE-SME)
│   ├── breakout_charts/          # rendered Plotly chart HTML
│   ├── _cache_ohlcv/             # legacy scanner OHLCV cache
│   ├── multi_pct_down_report.xlsx  # input universe for the scanner
│   └── breakout_watchlist.xlsx   # latest scanner output
│
├── upper_band/                   # NSE upper-circuit deep-dive pipeline
│   ├── upperband_analyze.py
│   ├── charts/<DATE>/
│   ├── analysis/<DATE>/
│   └── Upper Band.csv
│
├── launchd/
│   └── com.mlbreakouts.dailyscan.plist  # macOS scheduler config
│
├── notebooks/                    # exploratory work (gitignored)
├── tests/
├── Output/                       # daily artefacts (see § 5)
├── logs_*.txt                    # historical phase / optuna run logs
├── README.md
└── requirements.txt
```

`configs/default.yaml` is the **single source of truth** for the ML
pipeline. The legacy scanner has its own thresholds embedded in
`legacy_scanner/breakout_scanner_angel.py` (v3.5 HC rule:
`pocket_pivot AND lvs >= 0.4 AND base_tight`).

---

## 4. The two pipelines

### 4.1 ML pipeline (training + inference)

**Setup detector → features → meta-model → calibrated probability.**

| Stage | What it does |
|---|---|
| Primary | `src/setup_detector.py` — fractal-pivot resistance detection + base-formation gates (config: `setup` block). Hard gates eliminate ~99% of bars. |
| Features | `src/features.py` — ~110 columns: price/volume, sector RS, regime, fundamentals (`fund_*` + cross-sectional ranks), volatility, base geometry. Inputs: bench, sector indices, fundamentals. |
| Labels | `src/labeling.py` — **triple-barrier** (`+3 ATR / -1 ATR / 10d`), plus `require_close_above_R` confirmation in first 5 bars. Sample-uniqueness weights from label overlap. |
| CV | `src/cv.py` — Walk-forward (default; expanding window) or PurgedKFold with embargo. **Last 12 months are strict holdout** never seen in training. |
| Model | `src/model.py` — LightGBM with monotone constraints + isotonic calibration. Optional **stacker** (LGBM+XGB-DART → LR meta-learner), **regime-conditioned** sub-models (split by `bench_above_200dma`), **R-multiple regressor** for Kelly sizing. |
| Inference | At each bar: detect setup → make_features → predict → blend regime/stacker → calibrate → threshold (`watchlist 0.15`, `trigger 0.22`). |

### 4.2 Legacy scanner (rule-only, pre-breakout)

`legacy_scanner/breakout_scanner_angel.py` reads
`multi_pct_down_report.xlsx` (output of `multi_pct_down.py`), pulls
OHLCV via Angel, and applies hard gates + pattern detectors.

**v3.5 HC rule**: `pocket_pivot AND lvs >= 0.4 AND base_tight`
(`base_tight` = `range_pct ≤ 0.15 OR trailing_pct ≤ 0.08`).

**Pattern detectors**: pocket_pivot (the only filter with measurable
edge on n=247 backtested signals — 50% wr solo), TTM squeeze, Wyckoff
spring, OBV divergence, flat-base flag, gap-fill flag. Cup-and-handle
and VCP detectors are known-broken and excluded from HC promotion.

Calibrated against an in-tree backtest (`backtest_breakouts.py`,
`analyze_signal_combos.py`) over 71 as-of dates × 18 months. 90%
precision is statistically unreachable on the available signal density;
v3.5 produces ~0–2 HC + 1–5 WL per day.

---

## 5. Outputs

All daily artefacts live in `Output/`:

| File | Source | Description |
|---|---|---|
| `scan_<YYYYMMDD>.csv` | `05_daily_scan.py` | Full ranked watchlist (`prob >= watchlist_thr`) |
| `triggers_<YYYYMMDD>.csv` | `05_daily_scan.py` | High-conviction subset (`prob >= trigger_thr`) |
| `sector_rotation_<YYYYMMDD>.csv` | `08_sector_rotation.py` | GICS sector leaderboard |
| `sector_rotation.parquet` | `08_sector_rotation.py` | Always-latest sector snapshot |
| `theme_rotation_<YYYYMMDD>.csv` | `08_sector_rotation.py` | Custom-theme leaderboard |
| `theme_rotation.parquet` | `08_sector_rotation.py` | Always-latest theme snapshot |
| `watchlist_in_leaders_<YYYYMMDD>.csv` | `08_sector_rotation.py` | ML setups in top-N sectors |
| `watchlist_in_theme_leaders_<YYYYMMDD>.csv` | `08_sector_rotation.py` | ML setups in top-N themes |
| `MLRotation_<YYYYMMDD>_<HHMMSS>.xlsx` | `08_sector_rotation.py` | Combined 8-sheet workbook |
| `logs/daily_scan_<TS>.log` | `run_daily_scan.sh` | Full launchd log (last 60 retained) |

Legacy scanner outputs land in `legacy_scanner/`:
`breakout_watchlist.xlsx`, `breakout_charts/<symbol>_breakout.html`.

Upper-circuit deep-dive outputs land in
`upper_band/analysis/<DATE>/` and `upper_band/charts/<DATE>/`.

---

## 6. What each parquet costs to refresh

| File | Rewritten by daily run? | Notes |
|---|---|---|
| `data/ohlcv/<SYM>.parquet` (~2500) | **Yes**, by default | Skip with `--no-refresh` (or `--no-update` for `05_daily_scan`). Bulk Angel pull dominates wall time (~10 min). |
| `data/benchmark_NIFTY.parquet` | **Yes**, by default | Skipped with `--no-refresh`. |
| `data/sector_indices.parquet` | **Yes** (always — `rebuild_indices=True` hard-coded) | |
| `data/theme_indices.parquet` | **Yes** (unless `--no-themes`) | |
| `data/fundamentals.parquet` | **No** | Only `06_fetch_fundamentals.py` writes it (quarterly cadence). |
| `data/training/setups.parquet` | **No** | Only `02_build_training_set.py` (quarterly). |
| `data/models/*.parquet` (`oof_predictions.parquet`) | **No** | Only `03_train_model.py` (quarterly). |

---

## 7. Operational cadence

| When | Command | Purpose |
|---|---|---|
| **Daily** (auto, launchd) | `bash scripts/run_daily_scan.sh` → `05_daily_scan.py` → chains `08_sector_rotation.py` | Production scan + sector + theme rotation + combined workbook |
| **Daily** (manual, parallel) | `python legacy_scanner/multi_pct_down.py` then `python legacy_scanner/breakout_scanner_angel.py` | Rule-only pre-breakout audit |
| **Daily** (optional) | `python upper_band/upperband_analyze.py` | Upper-circuit cohort study |
| **Quarterly refresh** | `06 → 01 → 02 → 03 → 04` | Refresh fundamentals, OHLCV history, training set, retrain, evaluate |
| **Yearly tune** | `07 → 03 → 04` | Optuna search → retrain with winners → re-evaluate |
| **Ad-hoc evaluation** | `python scripts/04_evaluate.py` | OOF metrics, decile, calibration after any retrain |

### launchd scheduling

```bash
cp launchd/com.mlbreakouts.dailyscan.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.mlbreakouts.dailyscan.plist
launchctl enable gui/$(id -u)/com.mlbreakouts.dailyscan
launchctl kickstart -k gui/$(id -u)/com.mlbreakouts.dailyscan
```

`scripts/run_daily_scan.sh` is the entry point: skips weekends
(`DOW >= 6`), tees output to `Output/logs/daily_scan_<TS>.log`,
rotates logs (keeps 60). Status:
```bash
launchctl print gui/$(id -u)/com.mlbreakouts.dailyscan | grep -E 'state|pid|runs|last exit code'
```

---

## 8. Quick start

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp legacy_scanner/.env.template legacy_scanner/.env  # fill Angel credentials

# 2. Bootstrap data (first time only)
python scripts/06_fetch_fundamentals.py    # fundamentals + sector idx
python scripts/01_build_dataset.py         # OHLCV cache (~10 min)

# 3. Train (or use existing data/models/)
python scripts/02_build_training_set.py    # replay history
python scripts/03_train_model.py           # train LGBM
python scripts/04_evaluate.py              # honest OOF metrics

# 4. Run daily scan
python scripts/05_daily_scan.py            # full pipeline
# or skip refresh for a fast re-run:
python scripts/05_daily_scan.py --no-update --rotation-no-refresh

# 5. (parallel) legacy scanner
python legacy_scanner/multi_pct_down.py
python legacy_scanner/breakout_scanner_angel.py
```

---

## 9. Why LightGBM (not Random Forest, not Transformers)

| Model | Verdict |
|---|---|
| Random Forest | Outdated since 2017. Loses to GBM on every tabular benchmark. |
| **LightGBM** | Industry standard for daily equity alpha. Used by WorldQuant, Two Sigma, Numerai, Kaggle JaneStreet/Optiver/JPX winners. |
| XGBoost | Equivalent to LightGBM, slightly slower. (Used as a stacker member.) |
| CatBoost | Strong alternative; disabled here (~30+ min/fold without GPU). |
| TabNet | Loses to LGBM in Shwartz-Ziv & Armon (NeurIPS 2022). |
| TFT / Transformer | Needs ≥50k samples and GPUs. We have ~24k setups. Overkill. |
| LSTM / GRU | Replaced by TCN/Transformer; both lose to LGBM on tabular data. |

LightGBM models `P(setup succeeds)`, **not** `next-day return`.

---

## 10. Known limitations & gotchas

- **`renewAccessToken()` SmartAPI bug**: occasional `signature` error
  during mid-run token refresh; auto-recovers.
- **OHLCV cache hygiene**: `data/ohlcv/` has ~2503 files but the live
  universe is ~1140 — older listings are never evicted.
- **`fundamentals.parquet` is not auto-refreshed**: sector mappings can
  drift. Re-run `scripts/06_fetch_fundamentals.py` quarterly.
- **`require_close_above_R` label v2**: tighter than v1; positives are
  rarer (~12% base rate) but more meaningful.
- **`scale_pos_weight: 0.9354`** intentionally <1 to penalize false
  positives in the imbalanced setting (Optuna chose this).
- **Legacy `cup_and_handle` + `vcp_contractions`**: detectors are known
  broken (0 / 16 wins; 7.7% wr respectively). Excluded from v3.5 HC.

---

## 11. References

- López de Prado, *Advances in Financial Machine Learning* (2018) —
  Ch. 3 (triple-barrier), Ch. 4 (sample weighting), Ch. 7 (purged CV)
- Microsoft Qlib — https://github.com/microsoft/qlib
- Shwartz-Ziv & Armon, *Tabular Data: Deep Learning is Not All You Need*
  (NeurIPS 2022)
- Numerai whitepaper — feature neutralization & meta-modeling
- Bensdorp, *Automated Stock Trading Systems*
- O'Neil, *How to Make Money in Stocks* (pocket-pivot definition used
  in legacy_scanner)




Details of when to run and train ML model, which file to run and cyclical duration:

<img width="2200" height="664" alt="image" src="https://github.com/user-attachments/assets/5a83ba76-6ef5-4fca-b224-c6fe220f5343" />


Details of parquet files and there run cycle:

<img width="2200" height="486" alt="image" src="https://github.com/user-attachments/assets/4698dae6-b9d3-437f-b499-e81560e9ee84" />


Details of all the parquet files in "data" folder, along with which python file generates which parquet file:

<img width="1854" height="1284" alt="image" src="https://github.com/user-attachments/assets/1f0bcac3-bd96-4beb-921e-b0bf2173846b" />



