"""Project paths + YAML config loader (single source of truth).

Purpose:
  Centralise every directory and config-file path used across the
  ML pipeline so each script imports the same constants and never
  hard-codes paths.

How it works:
  * `ROOT` is resolved from this file's location (one parent up from `src/`).
  * Each constant is a `pathlib.Path` derived from `ROOT`.
  * On import, the four "writable" dirs are auto-created
    (`OHLCV_DIR`, `TRAINING_DIR`, `MODELS_DIR`, `OUTPUT_DIR`).
  * `load_config(name)` returns the parsed YAML at
    `configs/<name>.yaml` (default = `default`).

Directory layout (relative to ROOT):
  data/ohlcv/        per-symbol OHLCV parquets (Angel One)
  data/training/     setups.parquet (labeled training corpus)
  data/models/       lgbm.txt, calibrator.pkl, features.json, ...
  data/<misc>        fundamentals.parquet, sector_indices.parquet,
                     benchmark_NIFTY.parquet, sectors_seed.json
  configs/           default.yaml (and overrides)
  Output/            daily run artefacts (scan / triggers / watchlist /
                     sector rotation / MLRotation_*.xlsx)

Data sources:
  None directly. Pure path / YAML utility module.

Outputs:
  Constants only; `load_config()` returns a dict.

How to run:
  Import-only. No CLI.
      from src.paths import OUTPUT_DIR, load_config
      cfg = load_config()           # configs/default.yaml

Notes:
  * `OUTPUT_DIR` lives at the project root (NOT inside `data/`) so
    that ad-hoc Excel/CSV outputs are easy to spot and ship.
  * `DATA_DIR` itself is intentionally NOT auto-created here — its
    presence is assumed (it ships with the repo).
"""

from __future__ import annotations
import os
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
OHLCV_DIR = DATA_DIR / "ohlcv"
TRAINING_DIR = DATA_DIR / "training"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = ROOT / "Output"  # daily scan results, sector reports, watchlists

for _d in (OHLCV_DIR, TRAINING_DIR, MODELS_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_config(name: str = "default") -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
