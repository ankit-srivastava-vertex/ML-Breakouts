"""Custom-theme rotation pipeline (parallel to src/sector_rotation.py).

Purpose:
  Mirror exactly what `src.sector_rotation` does, but operate on USER-DEFINED
  thematic baskets read from `src/index_constituents.json` (e.g. Wires&Cables,
  Forgings, Aerospace&Defense ...) instead of yfinance GICS sectors.

How it works:
  1. load_themes()                  → {theme: [symbols]} from JSON
  2. ensure_constituent_ohlcv()     → pull missing constituents via Angel One
                                      (delegates to sector_rotation.download_ohlcv)
  3. build_theme_indices()          → equal-weight cumulative-return index per
                                      theme (same algorithm as
                                      sector_rotation.build_sector_indices,
                                      but the symbol→theme map comes from JSON,
                                      not fundamentals)
  4. compute_rotation()             → reuses
                                      sector_rotation.compute_rotation() with
                                      a theme-specific sym_to_sec map for
                                      breadth.
  5. run_full_pipeline()            → orchestrates 1-4.

Why a parallel module (vs. extending sector_rotation)?
  Keeps the sector pipeline byte-identical to the previous (verified)
  behaviour. All theme-specific paths / parquets / output names are
  separate, so neither pipeline can corrupt the other.

Data sources:
  * src/index_constituents.json        custom theme universe (committed)
  * data/ohlcv/*.parquet               Angel One OHLCV cache (shared)
  * data/benchmark_NIFTY.parquet       ^NSEI benchmark (shared)

Outputs:
  * data/theme_indices.parquet         wide DF: index=DATE, cols=theme
  * Returned `rot` DF (consumed by scripts/08_sector_rotation.py)

How to run:
  Programmatically:
      from src.theme_rotation import run_full_pipeline, leaderboard
      rot = run_full_pipeline()
      leaderboard(rot, n=5)
  CLI: invoked from `scripts/08_sector_rotation.py` (alongside sectors).

Notes:
  * Theme baskets are intentionally narrow (4-15 symbols), so
    `min_members` defaults to 3 here (vs 10 for GICS sectors).
  * `breadth_above_*dma` is still computed; with small baskets it may be
    based on as few as 3-4 symbols — interpret with that in mind.
"""

from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Reuse the proven sector_rotation primitives.
from src import sector_rotation as sr

# ─── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
THEMES_PATH = ROOT / "src" / "index_constituents.json"
THEME_INDEX_PATH = sr.DATA_DIR / "theme_indices.parquet"
THEME_ROTATION_PATH = sr.OUTPUT_DIR / "theme_rotation.parquet"


# ─── 1. theme universe ────────────────────────────────────────────────────

def load_themes(path: Path = THEMES_PATH) -> dict[str, list[str]]:
    """Return {theme_name: [symbol, ...]} from index_constituents.json."""
    with open(path, "r") as f:
        raw = json.load(f)
    out: dict[str, list[str]] = {}
    for theme, payload in raw.items():
        syms = payload.get("constituents") if isinstance(payload, dict) else payload
        if not syms:
            continue
        out[theme] = sorted({s.upper().strip() for s in syms if s})
    return out


def all_constituents(themes: Optional[dict[str, list[str]]] = None) -> list[str]:
    if themes is None:
        themes = load_themes()
    s: set[str] = set()
    for syms in themes.values():
        s.update(syms)
    return sorted(s)


def symbol_to_theme_map(themes: Optional[dict[str, list[str]]] = None) -> dict[str, str]:
    """Reverse map. If a symbol appears in multiple themes the first wins;
    JSON file is curated to be mostly disjoint."""
    if themes is None:
        themes = load_themes()
    m: dict[str, str] = {}
    for theme, syms in themes.items():
        for s in syms:
            m.setdefault(s, theme)
    return m


# ─── 2. OHLCV backfill for missing constituents ───────────────────────────

def ensure_constituent_ohlcv(themes: Optional[dict[str, list[str]]] = None,
                             ohlcv_start: dt.date = dt.date(2018, 1, 1),
                             force_refresh: bool = False,
                             verbose: bool = True) -> int:
    """Make sure every theme constituent has data/ohlcv/<SYM>.parquet on disk.

    Returns the number of symbols downloaded.
    """
    syms = all_constituents(themes)
    if force_refresh:
        to_fetch = syms
    else:
        have = {p.stem.upper() for p in sr.OHLCV_DIR.glob("*.parquet")}
        to_fetch = sorted(set(syms) - have)
    if not to_fetch:
        if verbose:
            print(f"  themes: OHLCV cache complete ({len(syms)} constituents)")
        return 0
    if verbose:
        print(f"  themes: downloading {len(to_fetch)} missing constituents "
              f"from Angel One ...")
    sr.download_ohlcv(to_fetch, start=ohlcv_start, verbose=verbose)
    return len(to_fetch)


# ─── 3. theme indices (equal-weight) ──────────────────────────────────────

def build_theme_indices(themes: Optional[dict[str, list[str]]] = None,
                        min_members: int = 3,
                        save: bool = True) -> pd.DataFrame:
    """Equal-weight cumulative-return index per theme.

    Same algorithm as sector_rotation.build_sector_indices(); only the
    {group -> [symbols]} map differs.
    """
    if themes is None:
        themes = load_themes()

    out: dict[str, pd.Series] = {}
    for theme, syms in themes.items():
        if len(syms) < min_members:
            continue
        rets = []
        for s in syms:
            df = sr.load_symbol(s)
            if df is None or len(df) < 250:
                continue
            r = df["Close"].pct_change()
            rets.append(r)
        if len(rets) < min_members:
            continue
        ret_df = pd.concat(rets, axis=1).fillna(0.0)
        idx = (1.0 + ret_df.mean(axis=1)).cumprod() * 100
        out[theme] = idx

    if not out:
        raise RuntimeError(
            "No theme indices built — OHLCV cache missing for constituents?"
        )
    wide = pd.concat(out, axis=1)
    wide.index = pd.to_datetime(wide.index).normalize()
    if save:
        THEME_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(THEME_INDEX_PATH)
        print(f"  built {wide.shape[1]} theme indices, "
              f"{wide.index.min().date()} → {wide.index.max().date()}")
    return wide


def load_theme_indices() -> Optional[pd.DataFrame]:
    if not THEME_INDEX_PATH.exists():
        return None
    return pd.read_parquet(THEME_INDEX_PATH)


# ─── 4. rotation analytics (delegates to sector_rotation) ─────────────────

def compute_rotation(asof: Optional[pd.Timestamp] = None,
                     theme_idx: Optional[pd.DataFrame] = None,
                     bench: Optional[pd.Series] = None,
                     themes: Optional[dict[str, list[str]]] = None,
                     compute_breadth: bool = True) -> pd.DataFrame:
    """Theme rotation snapshot. Delegates to sector_rotation.compute_rotation()
    with a theme-specific symbol→group map for breadth computation."""
    if theme_idx is None:
        theme_idx = load_theme_indices()
    if theme_idx is None or theme_idx.empty:
        raise RuntimeError("No theme_indices.parquet — run build_theme_indices().")
    sym_to_theme = symbol_to_theme_map(themes)
    return sr.compute_rotation(
        asof=asof,
        sector_idx=theme_idx,
        bench=bench,
        compute_breadth=compute_breadth,
        sym_to_sec=sym_to_theme,
    )


def leaderboard(rot: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Theme leaderboard (same column set as sector leaderboard)."""
    cols = ["rank", "mom_score", "rs_1m", "rs_3m", "rs_6m",
            "rank_chg_5d", "rank_chg_20d",
            "breadth_above_50dma", "breadth_above_200dma"]
    cols = [c for c in cols if c in rot.columns]
    top = rot.head(n)[cols].round(2)
    bot = rot.tail(n)[cols].round(2)
    print(f"\n  ── TOP {n} THEMES (LEADERS) ──")
    print(top.to_string())
    print(f"\n  ── BOTTOM {n} THEMES (LAGGARDS) ──")
    print(bot.to_string())
    if "rank_chg_5d" in rot.columns:
        rotating_in = rot.nlargest(min(3, len(rot)), "rank_chg_5d")
        print("\n  ── THEMES ROTATING IN (rank improving fastest, 5d) ──")
        print(rotating_in[["rank", "rank_chg_5d", "rs_1m", "rs_3m"]].round(2).to_string())
    return top


# ─── 5. end-to-end orchestrator ───────────────────────────────────────────

def run_full_pipeline(rebuild_indices: bool = True,
                      force_refresh_ohlcv: bool = False,
                      ohlcv_start: dt.date = dt.date(2018, 1, 1),
                      asof: Optional[pd.Timestamp] = None,
                      min_members: int = 3) -> pd.DataFrame:
    """End-to-end theme rotation.

    Reuses cached benchmark / OHLCV downloaded by the sector pipeline.
    Only fetches the constituents that are NOT already in the cache
    (unless `force_refresh_ohlcv=True`).
    """
    themes = load_themes()
    print(f"  [themes/1/3] {len(themes)} themes loaded "
          f"({len(all_constituents(themes))} unique constituents)")

    ensure_constituent_ohlcv(themes,
                             ohlcv_start=ohlcv_start,
                             force_refresh=force_refresh_ohlcv)

    if rebuild_indices or not THEME_INDEX_PATH.exists():
        print("  [themes/2/3] Building theme indices ...")
        build_theme_indices(themes, min_members=min_members)
    else:
        print("  [themes/2/3] Using cached theme indices")

    bench = sr.load_benchmark()  # already refreshed by sector pipeline
    print("  [themes/3/3] Computing rotation snapshot ...")
    return compute_rotation(asof=asof, bench=bench, themes=themes)
