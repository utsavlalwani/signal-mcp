"""Walk-forward backtester.

Realized P&L (Sharpe, returns, drawdown) is computed by VectorBT and persisted
to an artifact for the human reviewer. Only signal-quality metadata (IC,
turnover, decile spread) is returned to the LLM. The split is enforced here at
the MCP boundary, not in prompts.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt


@dataclass
class SignalQualityMetadata:
    """The ONLY portion of the backtest result the LLM is allowed to see."""
    factor_name: str
    n_periods: int
    ic_mean: float
    ic_ir: float
    turnover_pct_per_month: float
    decile_spread_bps_per_month: float
    artifact_uri: str


@dataclass
class BacktestArtifact:
    """Full backtest result -- stored to disk, NOT returned to the LLM."""
    factor_name: str
    universe: list[str]
    train_window_months: int
    test_window_months: int
    period_starts: list[date] = field(default_factory=list)
    realized_returns_pct: list[float] = field(default_factory=list)
    ic_series: list[float] = field(default_factory=list)
    turnover_series: list[float] = field(default_factory=list)
    decile_spread_series: list[float] = field(default_factory=list)
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown_pct: float | None = None
    cumulative_return_pct: float | None = None


def _trading_month_ends(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    df = pd.DataFrame(index=idx)
    df["month"] = df.index.to_period("M")
    return list(df.groupby("month").apply(lambda g: g.index.max()))


def walk_forward_backtest(
    factor_name: str,
    factor_signal_at_month_end: dict[pd.Timestamp, pd.Series],
    prices: pd.DataFrame,
    train_window_months: int = 24,
    test_window_months: int = 1,
    long_short: bool = True,
    transaction_cost_bps: float = 10.0,
) -> BacktestArtifact:
    if not factor_signal_at_month_end:
        raise ValueError("No factor signals provided")

    month_ends = sorted(factor_signal_at_month_end.keys())
    if len(month_ends) < train_window_months + test_window_months:
        raise ValueError(
            f"Need at least {train_window_months + test_window_months} months of signal; "
            f"got {len(month_ends)}"
        )

    trading_month_ends = _trading_month_ends(prices.index)
    me_to_tme: dict[pd.Timestamp, pd.Timestamp] = {}
    for me in month_ends:
        candidates = [t for t in trading_month_ends if t <= me]
        if candidates:
            me_to_tme[me] = candidates[-1]

    universe = sorted({tk for sig in factor_signal_at_month_end.values() for tk in sig.index.tolist()})
    px = prices.reindex(columns=universe).ffill()

    # Daily target-weight matrix: NaN except on rebalance trading days (=> no order between).
    weights = pd.DataFrame(np.nan, index=px.index, columns=universe)

    period_starts: list[date] = []
    ic_series: list[float] = []
    turnover_series: list[float] = []
    decile_spread_series: list[float] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for i in range(train_window_months, len(month_ends) - test_window_months + 1):
        rebal_me = month_ends[i]
        end_idx = i + test_window_months - 1
        end_me = month_ends[end_idx] if end_idx < len(month_ends) else None
        if end_me is None or rebal_me not in me_to_tme or end_me not in me_to_tme:
            continue

        signal = factor_signal_at_month_end[rebal_me].dropna()
        if signal.empty or len(signal) < 10:
            continue
        try:
            deciles = pd.qcut(signal, 10, labels=False, duplicates="drop")
        except ValueError:
            continue

        top = set(signal[deciles == deciles.max()].index.tolist())
        bot = set(signal[deciles == 0].index.tolist())

        t_start = me_to_tme[rebal_me]
        t_end = me_to_tme[end_me]
        fwd_ret = (px.loc[t_end] / px.loc[t_start] - 1.0).dropna()

        common = signal.index.intersection(fwd_ret.index)
        if len(common) < 5:
            continue
        ic = signal.loc[common].rank().corr(fwd_ret.loc[common].rank())

        top_ret = fwd_ret.reindex(list(top)).mean() if top else 0.0
        bot_ret = fwd_ret.reindex(list(bot)).mean() if bot else 0.0
        decile_spread_bps = (top_ret - bot_ret) * 10000.0

        if prev_long or prev_short:
            churn = len(top.symmetric_difference(prev_long)) + len(bot.symmetric_difference(prev_short))
            denom = max(len(top) + len(bot), 1)
            turnover = churn / (2.0 * denom)
        else:
            turnover = 1.0

        # Set target weights on this rebalance trading day.
        if top:
            weights.loc[t_start, list(top)] = 0.5 / len(top)
        if long_short and bot:
            weights.loc[t_start, list(bot)] = -0.5 / len(bot)

        period_starts.append(rebal_me.date())
        ic_series.append(float(ic) if not np.isnan(ic) else 0.0)
        turnover_series.append(float(turnover))
        decile_spread_series.append(float(decile_spread_bps))
        prev_long, prev_short = top, bot

    if not period_starts:
        raise RuntimeError("Walk-forward produced no valid periods")

    # Realized P&L via VectorBT (this is the LLM-withheld half).
    pf = vbt.Portfolio.from_orders(
        px,
        size=weights,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        fees=transaction_cost_bps / 10000.0,
        freq="1D",
    )
    daily_ret = pf.returns()
    monthly_ret = daily_ret.resample("ME").apply(lambda s: (1.0 + s).prod() - 1.0)
    realized_returns_pct = [float(r) * 100.0 for r in monthly_ret.values]

    sharpe = float(pf.sharpe_ratio()) if np.isfinite(pf.sharpe_ratio()) else None
    sortino = float(pf.sortino_ratio()) if np.isfinite(pf.sortino_ratio()) else None
    max_dd = float(pf.max_drawdown()) * 100.0
    cum = float(pf.total_return()) * 100.0

    return BacktestArtifact(
        factor_name=factor_name,
        universe=universe,
        train_window_months=train_window_months,
        test_window_months=test_window_months,
        period_starts=period_starts,
        realized_returns_pct=realized_returns_pct,
        ic_series=ic_series,
        turnover_series=turnover_series,
        decile_spread_series=decile_spread_series,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd,
        cumulative_return_pct=cum,
    )


ARTIFACT_DIR = Path("data/backtests")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def persist_artifact(artifact: BacktestArtifact) -> str:
    artifact_id = uuid.uuid4().hex[:12]
    uri = f"quant://backtests/{artifact_id}"
    path = ARTIFACT_DIR / f"{artifact_id}.json"
    payload = asdict(artifact)
    payload["period_starts"] = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d in payload["period_starts"]]
    path.write_text(json.dumps(payload, indent=2))
    return uri


def to_signal_quality_metadata(artifact: BacktestArtifact, uri: str) -> SignalQualityMetadata:
    ic_arr = np.array(artifact.ic_series) if artifact.ic_series else np.array([0.0])
    return SignalQualityMetadata(
        factor_name=artifact.factor_name,
        n_periods=len(artifact.period_starts),
        ic_mean=float(ic_arr.mean()),
        ic_ir=float(ic_arr.mean() / ic_arr.std(ddof=0)) if ic_arr.std(ddof=0) > 0 else 0.0,
        turnover_pct_per_month=float(np.mean(artifact.turnover_series) * 100.0) if artifact.turnover_series else 0.0,
        decile_spread_bps_per_month=float(np.mean(artifact.decile_spread_series)) if artifact.decile_spread_series else 0.0,
        artifact_uri=uri,
    )
