"""Factor library -- 5 standard factors implemented honestly.

Every factor here is well-known from the academic literature. No claims of proprietary edge. These are reference implementations meant to demonstrate the *infrastructure*, not to make money.

    * value         -- book-to-market (B/P)
    * momentum      -- 12-1 month price momentum
    * quality       -- return on equity (ROE)
    * low_vol       -- inverse 252-day realized volatility
    * residual_mom  -- momentum residualized against the equal-weight market
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorDef:
    name: str
    description: str
    direction: int  # +1 long top decile, -1 long bottom decile
    needs_fundamentals: bool


FACTOR_DEFS: dict[str, FactorDef] = {
    "value": FactorDef(
        name="value",
        description="Book-to-market ratio. High = cheap.",
        direction=+1,
        needs_fundamentals=True,
    ),
    "momentum": FactorDef(
        name="momentum",
        description="12-month price return skipping the last month (12-1).",
        direction=+1,
        needs_fundamentals=False,
    ),
    "quality": FactorDef(
        name="quality",
        description="Return on equity (net income / equity).",
        direction=+1,
        needs_fundamentals=True,
    ),
    "low_vol": FactorDef(
        name="low_vol",
        description="Inverse of 252-day realized return volatility. Lower vol = higher score.",
        direction=+1,
        needs_fundamentals=False,
    ),
    "residual_mom": FactorDef(
        name="residual_mom",
        description="Momentum residualized against the equal-weight universe return.",
        direction=+1,
        needs_fundamentals=False,
    ),
}


def list_factors() -> list[dict]:
    return [
        {
            "name": f.name,
            "description": f.description,
            "direction": f.direction,
            "needs_fundamentals": f.needs_fundamentals,
        }
        for f in FACTOR_DEFS.values()
    ]


# --- Factor computational kernels ----------------------------------------
# Each takes price panel (and fundamentals where needed), returns Series[ticker -> score].


def factor_momentum(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """12-1 month momentum: t-12m to t-1m total return per ticker."""
    asof_idx = prices.index.searchsorted(asof, side="right") - 1
    if asof_idx < 252:
        raise ValueError("Need at least 252 trading days of history")
    end_idx = asof_idx - 21
    start_idx = asof_idx - 252
    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError("Insufficient history")
    p_start = prices.iloc[start_idx]
    p_end = prices.iloc[end_idx]
    return ((p_end / p_start) - 1.0).dropna()


def factor_low_vol(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Inverse 252-day realized volatility of daily log returns."""
    asof_idx = prices.index.searchsorted(asof, side="right") - 1
    if asof_idx < 252:
        raise ValueError("Need at least 252 trading days of history")
    window = prices.iloc[asof_idx - 252:asof_idx]
    rets = np.log(window / window.shift(1)).dropna(how="all")
    vol = rets.std() * np.sqrt(252.0)
    inv_vol = 1.0 / vol.replace(0, np.nan)
    return inv_vol.dropna()


def factor_residual_mom(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Momentum residualized against equal-weight market."""
    mom = factor_momentum(prices, asof)
    market = mom.mean()
    return mom - market


def factor_value(fundamentals: dict, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Book-to-market: total_equity / market_cap.
    `fundamentals` is a dict of ticker -> {total_equity, shares_outstanding, ...}.
    """
    asof_idx = prices.index.searchsorted(asof, side="right") - 1
    last_prices = prices.iloc[asof_idx]
    out = {}
    for tk, f in fundamentals.items():
        if not f or f.get("total_equity") is None or f.get("shares_outstanding") in (None, 0):
            continue
        try:
            mkt_cap = float(last_prices[tk]) * float(f["shares_outstanding"])
            if mkt_cap <= 0:
                continue
            out[tk] = float(f["total_equity"]) / mkt_cap
        except (KeyError, ValueError, TypeError):
            continue
    return pd.Series(out).dropna()


def factor_quality(fundamentals: dict) -> pd.Series:
    """Return on equity = net_income_ttm / total_equity."""
    out = {}
    for tk, f in fundamentals.items():
        if not f:
            continue
        ni = f.get("net_income_ttm")
        eq = f.get("total_equity")
        if ni is None or eq is None or eq <= 0:
            continue
        out[tk] = float(ni) / float(eq)
    return pd.Series(out).dropna()


def compute_factor(
    name: str,
    prices: pd.DataFrame,
    asof: pd.Timestamp,
    fundamentals: dict | None = None,
) -> pd.Series:
    """Dispatch to the right kernel and return a cross-sectional Z-score."""
    defn = FACTOR_DEFS.get(name)
    if defn is None:
        raise ValueError(f"Unknown factor: {name}. Available: {list(FACTOR_DEFS)}")

    if name == "momentum":
        raw = factor_momentum(prices, asof)
    elif name == "low_vol":
        raw = factor_low_vol(prices, asof)
    elif name == "residual_mom":
        raw = factor_residual_mom(prices, asof)
    elif name == "value":
        if not fundamentals:
            raise ValueError("Value factor requires fundamentals")
        raw = factor_value(fundamentals, prices, asof)
    elif name == "quality":
        if not fundamentals:
            raise ValueError("Quality factor requires fundamentals")
        raw = factor_quality(fundamentals)
    else:
        raise ValueError(f"Factor '{name}' has no kernel registered")

    # Direction-aware Z-score so larger = better
    z = (raw - raw.mean() / raw.std(ddof=0) if raw.std(ddof=0) > 0 else raw - raw.mean())
    return (z * defn.direction).rename(name)
