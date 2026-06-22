"""Quant Engine MCP Server.

Hosts the heaviest tools: factor library, walk-forward backtesting, and portfolio optimization. This is where the agent-blind eval is enforced.

Port: 8004
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import date, datetime, timedelta
from typing import Annotated, Literal

import httpx
import numpy as np
import pandas as pd
from fastmcp import FastMCP, Context
from fastmcp.server.elicitation import AcceptedElicitation
from pydantic import BaseModel, Field

from agent.config import settings
from agent.tracing import init_tracing, get_tracer, continue_trace_from_meta
from servers._auth import build_verifier

from servers.quant_engine.factor_library import (
    list_factors as _list_factors,
    compute_factor as _compute_factor,
    FACTOR_DEFS,
)
from servers.quant_engine.backtest import (
    walk_forward_backtest as _wfb,
    persist_artifact,
    to_signal_quality_metadata,
    BacktestArtifact,
    SignalQualityMetadata
)
from servers.quant_engine.optimizer import optimize as _optimize, OptObjective


init_tracing("quant_engine")
tracer = get_tracer(__name__)


mcp = FastMCP(
    name="quant-engine",
    instructions=(
        "Quant-research engine. Compute factors, run walk-forward backtests, "
        "and optimize portfolios. IMPORTANT: backtest tools return only signal-"
        "quality metadata (IC, turnover, decile spread). Realized P&L and "
        "Sharpe are deliberately withheld to prevent metric gaming."
    ),
    auth=build_verifier(),
)


# --- Schemas ----------------------------------------------------------


class FactorListItem(BaseModel):
    name: str
    description: str
    direction: str
    needs_fundamentals: bool


class FactorExposure(BaseModel):
    ticker: str
    z_score: float


class FactorPanel(BaseModel):
    factor_name: str
    asof_date: date
    universe_size: int
    exposures: list[FactorExposure]


class WalkForwardRequest(BaseModel):
    factor_name: str
    universe_name: str = "demo-50"
    start: str  # YYYY-MM-DD
    end: str
    train_window_months: int = 24
    test_window_months: int = 1
    long_short: bool = True
    transaction_cost_bps: float = 10.0


class OptimizationRequest(BaseModel):
    tickers: list[str]
    start: str
    end: str
    objective: OptObjective = "hrp"
    max_weight: float = 0.10


class SweepRequest(BaseModel):
    factor_name: str
    universe_name: str
    start: str
    end: str
    train_windows: list[int] = Field(default_factory=lambda: [12, 24, 36])
    transaction_costs_bps: list[float] = Field(default_factory=lambda: [5.0, 10.0])
    test_window_months: int = 1
    long_short: bool = True


# --- Helper: pull OHLCV from the market-data MCP server


async def _fetch_prices(
    tickers: list[str], start: date, end: date
) -> pd.DataFrame:
    """Pull adjusted close prices from market-data-mcp into a wide DataFrame."""
    base = f"http://localhost:{settings.market_data_port}/mcp"
    panels: dict[str, pd.Series] = {}

    # We talk to market-data via direct httpx calls to its underlying yfinance path here for speed (the MCP layer would round-trip through JSON-RPC; for an internal server-to-server data pull, that overhead isn't justified). In a stricter design, you'd use a MultiServerMCPClient here too.

    import yfinance as yf

    df = yf.download(
        tickers,
        start=start,
        end=end + timedelta(days=1),
        progress=False,
        auto_adjust=False,
        group_by="ticker",
    )

    if df.empty:
        raise RuntimeError("No price data returned")

    out: dict[str, pd.Series] = {}
    for tk in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                series = df[tk]["Adj Close"].dropna()
            else:
                series = df["Adj Close"].dropna()
            if not series.empty:
                out[tk] = series
        except (KeyError, AttributeError):
            continue

    if not out:
        raise RuntimeError("No valid price series after parsing")

    return pd.DataFrame(out).sort_index()


# Hard-coded demo-50 (mirrors market_data served; in prod, fetch from market-data MCP)
_DEMO_50 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "LLY", "ABBV", "UNH", "V", "ХОМ", "MA", "GOOGL", "META", "TSLA", "TMO", "CVX", "WMT", "BAC", "KO", "PG", "JNJ", "HD", "COST", "MRK", "BRK-B", "AVGO", "JPM", "ACN", "ABT", "NFLX", "PEP", "CRM", "ORCL", "ADBE", "MCD", "VZ", "AMD", "INTC", "QCOM", "LIN", "DHR", "WFC", "TXN", "DIS", "INTU", "CSCO", "PFE", "CMCSA", "PM", "BMY", "IBM",
]


def _resolve_universe(name: str) -> list[str]:
    n = name.lower()
    if n in {"demo-50", "demo50", "demo"}:
        return _DEMO_50
    if n in {"demo-10", "demo10"}:
        return _DEMO_50[:10]
    raise ValueError(f"Unknown universe: {name}")


# --- Tools ----------------------------------------------------------\


@ mcp.tool()
async def list_factor(ctx: Context) -> list[FactorListItem]:
    """Available factors with their descriptions and conventions."""
    with continue_trace_from_meta(getattr((ctx, "meta", None))):
        with tracer.start_as_current_span("list_factors"):
            return [FactorListItem(**f) for f in _list_factors()]


@mcp.tool()
async def compute_factor(
    factor_name: Annotated[str, "Factor name for list_factors"],
    universe_name: Annotated[str, "Universe: demo-50 or demo-10"],
    asof_date: Annotated[str, "Date YYYY-MM-DD"],
    ctx: Context,
) -> FactorPanel:
    """Compute cross-sectional Z-score for the requested factor."""
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("compute_factor") as span:
            span.set_attributes({"factor": factor_name, "universe": universe_name, "asof": asof_date})
            tickers = _resolve_universe(universe_name)
            asof = pd.Timestamp(asof_date)

            # Pull ~14 months of history (factor kernels need up to 252 trading days)
            start = (asof - pd.Timedelta(days=400)).date()
            prices = await _fetch_prices(tickers, start, asof.date())

            fundamentals = None
            if FACTOR_DEFS[factor_name].needs_fundamentals:
                # For the demo, we stub fundamentals from the latest market cap snapshot to avoid a heavy EDGAR XBRL fan-out per backtest period.
                # In production, swap in a real call to edgar-fundamentals MCP.
                fundamentals = _stub_fundamentals(tickers, prices, asof)

            scores = _compute_factor(factor_name, prices, asof, fundamentals)
            return FactorPanel(
                factor_name=factor_name,
                asof_date=asof.date(),
                universe_size=len(scores),
                exposures=[FactorExposure(ticker=tk, z_score=float(v)) for tk, v in scores.items()],
            )


def _stub_fundamentals(tickers: list[str], prices: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """Deterministic fundamentals stub for demo purposes.

    Uses the ticker hash to fabricate stable but distinguishable values. Marked as a stub on purpose -- switch this to a real EDGAR MCP call when ready.
    """
    out = {}
    for tk in tickers:
        if tk not in prices.columns:
            continue
        seed = sum(ord(c) for c in tk)
        rng = np.random.default_rng(seed)
        out[tk] = {
            "net_income_ttm": rng.uniform(1e9, 1e11),
            "total_equity": rng.uniform(5e9, 5e11),
            "shares_outstanding": rng.uniform(5e8, 1.5e10),
        }
    return out


@mcp.tool()
async def factor_ic(
    factor_name: str,
    universe_name: str,
    start: Annotated[str, "Start YYYY-MM-DD"],
    end: Annotated[str, "End YYYY-MM-DD"],
    horizon_days: Annotated[int, "Forward return horizon"] = 21,
    ctx: Context = None,
) -> dict:
    """Information Coefficient time series for a factor."""
    with continue_trace_from_meta(getattr(ctx, "meta", None) if ctx else None):
        with tracer.start_as_current_span("factor_ic") as span:
            span.set_attributes({"factor": factor_name})

            tickers = _resolve_universe(universe_name)
            start_d = date.fromisoformat(start)
            end_d = date.fromisoformat(end)
            prices = await _fetch_prices(tickers, start_d - timedelta(days=400), end_d)

            fundamentals = _stub_fundamentals(tickers, prices, pd.Timestamp(end_d))

            month_ends = sorted({pd.Timestamp(d) for d in prices.index if d.is_month_end})
            ic_points = []
            for me in month_ends:
                if me < pd.Timestamp(start_d):
                    continue
                try:
                    scores = _compute_factor(factor_name, prices, me, fundamentals)
                    fwd_idx = prices.index.searchsorted(me) + horizon_days
                    if fwd_idx >= len(prices):
                        continue
                    fwd_ret = (prices.iloc[fwd_idx] / prices.loc[me] - 1.0).dropna()
                    common = scores.index.intersection(fwd_ret.index)
                    if len(common) >= 5:
                        ic = scores.loc[common].rank().corr(fwd_ret.loc[common].rank())
                        if not np.isnan(ic):
                            ic_points.append({"date": me.date().isoformat(), "ic": float(ic)})
                except Exception:
                    continue

            if not ic_points:
                return {"factor": factor_name, "ic_mean": 0.0, "n": 0, "points": []}
            ics = [p["ic"] for p in ic_points]
            return {
                "factor": factor_name,
                "n_periods": len(ics),
                "ic_mean": float(np.mean(ics)),
                "ic_std": float(np.std(ics, ddof=0)),
                "ic_ir": float(np.mean(ics) / np.std(ics, ddof=0)) if np.std(ics, ddof=0) > 0 else 0.0,
                "points": ic_points,
            }


async def _run_backtest(
    factor_name: str,
    universe_name: str,
    start: str,
    end: str,
    train_window_months: int,
    test_window_months: int,
    long_short: bool,
    transaction_cost_bps: float,
) -> tuple[SignalQualityMetadata, str]:
    """Resolve the universe, fetch prices, build the monthly signal panel, run the
    walk-forward backtest, persist the artifact, and return (metadata, artifact_uri)."""
    tickers = _resolve_universe(universe_name)
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    prices = await _fetch_prices(tickers, start_d - timedelta(days=400), end_d)
    fundamentals = _stub_fundamentals(tickers, prices, pd.Timestamp(end_d))

    # Precompute monthly factor signal
    month_ends = sorted({pd.Timestamp(d) for d in prices.index if d.is_month_end})
    signal_panel: dict[pd.Timestamp, pd.Series] = {}
    for me in month_ends:
        try:
            signal_panel[me] = _compute_factor(factor_name, prices, me, fundamentals)
        except Exception:
            continue

    artifact = _wfb(
        factor_name=factor_name,
        factor_signal_at_month_end=signal_panel,
        prices=prices,
        train_window_months=train_window_months,
        test_window_months=test_window_months,
        long_short=long_short,
        transaction_cost_bps=transaction_cost_bps,
    )
    uri = persist_artifact(artifact)
    return to_signal_quality_metadata(artifact, uri), uri


@mcp.tool()
async def walk_forward_backtest(
    request: WalkForwardRequest,
    ctx: Context,
) -> SignalQualityMetadata:
    """
    Walk-forward backtest of a factor.

    Returns ONLY signal-quality metadata to the agent (IC, turnover, decile spread). Realized P&L, Sharpe, and the full equity curve are persisted to an artifact at `artifact_uri` for human review -- the LLM cannot retrieve them.
    """
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("walk_forward_backtest") as span:
            span.set_attributes({"factor": request.factor_name, "universe": request.universe_name})
            metadata, _uri = await _run_backtest(
                factor_name=request.factor_name,
                universe_name=request.universe_name,
                start=request.start,
                end=request.end,
                train_window_months=request.train_window_months,
                test_window_months=request.test_window_months,
                long_short=request.long_short,
                transaction_cost_bps=request.transaction_cost_bps,
            )
            return metadata


@mcp.tool()
async def walk_forward_sweep(request: SweepRequest, ctx: Context) -> dict:
    """Grid sweep of walk-forward backtests over train-window x cost combos.

    Gated by an MCP elicitation prompt: a full sweep is expensive, so the
    client/user must confirm before it runs. Returns agent-blind metadata per
    combo (realized P&L stays withheld)."""
    combos = [(tw, c) for tw in request.train_windows for c in request.transaction_costs_bps]
    confirm = await ctx.elicit(
        f"walk_forward_sweep will run {len(combos)} backtests for factor "
        f"'{request.factor_name}' on universe '{request.universe_name}'. Proceed?",
        response_type=None,
    )
    if not isinstance(confirm, AcceptedElicitation):
        return {"status": "declined", "n_combos": len(combos), "results": []}

    results = []
    for tw, cost in combos:
        meta, uri = await _run_backtest(
            factor_name=request.factor_name,
            universe_name=request.universe_name,
            start=request.start,
            end=request.end,
            train_window_months=tw,
            test_window_months=request.test_window_months,
            long_short=request.long_short,
            transaction_cost_bps=cost,
        )
        row = asdict(meta)
        row["train_window_months"] = tw
        row["transaction_cost_bps"] = cost
        results.append(row)
    return {"status": "ok", "n_combos": len(combos), "results": results}


@mcp.tool()
async def compute_signal_decile_spread(
    factor_name: str,
    universe_name: str,
    asof_date: Annotated[str, "Date YYYY-MM-DD"],
    horizon_days: Annotated[int, "Forward return horizon"] = 21,
    ctx: Context = None,
) -> dict:
    """Long-short decile spread (top - bottom decile forward return) on a date."""
    with continue_trace_from_meta(getattr(ctx, "meta", None) if ctx else None):
        with tracer.start_as_current_span("compute_signal_decile_spread"):
            tickers = _resolve_universe(universe_name)
            asof = pd.Timestamp(asof_date)
            prices = await _fetch_prices(tickers, (asof - pd.Timedelta(days=400)).date(), asof.date() + timedelta(days=horizon_days + 5))
            fundamentals = _stub_fundamentals(tickers, prices, asof)
            scores = _compute_factor(factor_name, prices, asof, fundamentals)

            fwd_idx = prices.index.searchsorted(asof) + horizon_days
            if fwd_idx >= len(prices):
                return {"top_pct": None, "bot_pct": None, "spread_bps": None}
            fwd = (prices.iloc[fwd_idx] / prices.loc[asof] - 1.0).dropna()

            deciles = pd.qcut(scores, 10, labels=False, duplicates="drop")
            top = scores[deciles == deciles.max()].index
            bot = scores[deciles == 0].index
            top_ret = float(fwd.reindex(top).mean())
            bot_ret = float(fwd.reindex(bot).mean())
            return {
                "asof_date": asof_date,
                "horizon_days": horizon_days,
                "top_decile_ret_pct": top_ret * 100.0,
                "bot_decile_ret_pct": bot_ret * 100.0,
                "spread_bps": (top_ret - bot_ret) * 10000.0,
            }


@mcp.tool()
async def optimize_portfolio(
    request: OptimizationRequest,
    ctx: Context,
) -> dict:
    """Portfolio optimization: HRP max-Sharpe, or min-CVaR."""
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("optimize_portfolio") as span:
            span.set_attributes({"objective": request.objective, "n_assets": len(request.tickers)})
            start_d = date.fromisoformat(request.start)
            end_d = date.fromisoformat(request.end)
            prices = await _fetch_prices(request.tickers, start_d, end_d)
            weights = _optimize(
                prices=prices,
                objective=request.objective,
                max_weight=request.max_weight,
            )
            return {
                "objective": request.objective,
                "n_assets": len(weights),
                "weights": weights,
                "sum_weights": round(sum(weights.values()), 4),
            }


@mcp.tool()
async def factor_turnover(
    factor_name: str,
    universe_name: str,
    start: str,
    end: str,
    ctx: Context,
) -> dict:
    """Per-month rebalance turnover for a factor (top + bottom deciles)."""
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("factor_turnover"):
            tickers = _resolve_universe(universe_name)
            start_d = date.fromisoformat(start)
            end_d = date.fromisoformat(end)
            prices = await _fetch_prices(tickers, start_d - timedelta(days=400), end_d)
            fundamentals = _stub_fundamentals(tickers, prices, pd.Timestamp(end_d))

            month_ends = sorted({pd.Timestamp(d) for d in prices.index if d.is_month_end and pd.Timestamp(start_d) <= d <= pd.Timestamp(end_d)})
            prev_top: set[str] = set()
            prev_bot: set[str] = set()
            turnovers: list[float] = []
            for me in month_ends:
                try:
                    scores = _compute_factor(factor_name, prices, me, fundamentals)
                    deciles = pd.qcut(scores, 10, labels=False, duplicates="drop")
                    top = set(scores[deciles == deciles.max()].index.tolist())
                    bot = set(scores[deciles == 0].index.tolist())
                    if prev_top or prev_bot:
                        churn = len(top.symmetric_difference(prev_top)) + len(bot.symmetric_difference(prev_bot))
                        denom = max(len(top) + len(bot), 1)
                        turnovers.append(churn / (2.0 * denom))
                    prev_top, prev_bot = top, bot
                except Exception:
                    continue

            return {
                "factor": factor_name,
                "n_periods": len(turnovers),
                "mean_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
                "median_turnover": float(np.median(turnovers)) if turnovers else 0.0
            }


@mcp.tool()
async def signal_quality_metadata(
    artifact_uri: Annotated[str, "URI returned by walk_forward_backtest"],
    ctx: Context,
) -> dict:
    """Re-fetch the LLM-visible portion of a backtest artifact.

    Same data as the original `walk_forward_backtest` response, exposed as a separate tool so the agent can revisit its own work. Realized P&L stays hidden.
    """
    import json
    from pathlib import Path
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("signal_quality_metadata"):
            artifact_id = artifact_uri.rsplit("/", 1)[-1]
            path = Path("data/backtests") / f"{artifact_id}.json"
            if not path.exists():
                raise FileNotFoundError(f"Artifact not found: {artifact_uri}")
            data = json.loads(path.read_text())
            ic = data.get("ic_series", []) or [0.0]
            tn = data.get("turnover_series", []) or [0.0]
            ds = data.get("decile_spread_series", []) or [0.0]

            return {
                "factor_name": data["factor_name"],
                "n_periods": len(data.get("period_starts", [])),
                "ic_mean": float(np.mean(ic)),
                "ic_ir": float(np.mean(ic) / np.std(ic, ddof=0)) if np.std(ic, ddof=0) > 0 else 0.0,
                "turnover_pct_per_month": float(np.mean(tn) * 100.0),
                "decile_spread_bps_per_month": float(np.mean(ds)),
                "artifact_uri": artifact_uri,
             }

if __name__ == '__main__':
    mcp.run(transport="streamable-http", host="0.0.0.0", port=settings.quant_engine_port)

