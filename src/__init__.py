"""ML Breakouts — meta-model for horizontal-resistance breakout setups.

Package marker for the `src/` library. Exposes `__version__`. All real
functionality lives in the sibling modules and is consumed by the
top-level numbered scripts under `scripts/`:

  paths            project paths + YAML config loader (single source of truth)
  cv               purged-k-fold + walk-forward CV (López de Prado, Ch. 7)
  data_ingestion   universe derivation from the OHLCV cache, ^NSEI fetch
  yf_ingestion     OHLCV downloader (Angel One via legacy_scanner.data_provider)
  fundamentals     yfinance .info cache (sector / mcap / margins ...)
  sectors          equal-weight sector indices + sym→sector map
  setup_detector   rule-based PRIMARY model (horizontal-resistance breakouts)
  features         ~110 ML features (geometry / vol / momentum / RS / fundas)
  labeling         triple-barrier labels + ATR helpers
  model            LightGBM trainer + isotonic calibrator + (optional) stacker
  sector_rotation  self-contained sector-rotation pipeline (Angel-only)
  universe         thin wrapper over data_ingestion.derive_universe_*

Usage:
    from src.paths import OUTPUT_DIR, MODELS_DIR
    from src.model import predict

No runtime side-effects on import; `src.paths` is the one module that
creates filesystem directories on first use.
"""
__version__ = "0.1.0"
