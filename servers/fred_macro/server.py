"""FRED Macro MCP Server.

St. Louis Fed economic data + a deterministic regime classifier built on classic rules (Sahm rule, yield curve, credit spreads). Deterministic -- so that it's reproducible across runs.

Port: 8003
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated, Literal

import pandas as pd
from fastmcp import FastMCP, Context
from fredapi import Fred
from pydantic import BaseModel, Field

from agent.config import settings
from agent.tracing import init_tracing, get_tracer, continue_trace_from_meta
from servers._auth import build_verifier


init_tracing("fred_macro")
tracer = get_tracer(__name__)


mcp = FastMCP(
    name="fred-macro",
    instructions=(
        "FRED macroeconomic data tools. Use `get_series` for any FRED series "
        "by its FRED code. Use `regime_classifier` for a deterministic regime "
        "label (expansion / slowdown / recession / recovery) on a given date."
    ),
    auth=build_verifier(),
)

if not settings.fred_api_key:
    # We don't hard-fail at import -- the server still starts so health-checks work. Individual tool calls raise a clear error if the key is missing.
    print("[fred_score] WARNING: FRED_API_KEY not set; tool calls will fail.")

_fred: Fred | None = None


def _get_fred() -> Fred:
    global _fred
    if _fred is None:
        if not settings.fred_api_key:
            raise RuntimeError("FRED_API_KEY env var is not set")
        _fred = Fred(api_key=settings.fred_api_key)
    return _fred


# --- Schemas ----------------------------------------------------


class SeriesPoint(BaseModel):
    date: date
    value: float


class SeriesResponse(BaseModel):
    series_id: str
    start: date
    end: date
    points: list[SeriesPoint]


RegimeLabel = Literal["expansion", "slowdown", "recession", "recovery", "unknown"]


class RegimeResponse(BaseModel):
    asof_date: date
    label: RegimeLabel
    unrate_3m_avg: float | None = None
    unrate_12m_min: float | None = None
    sahm_indicator: float | None = Field(
        None,
        description="Sahm rule: 3m-avg unemployment minus 12m-min. >= 0.5 historically marks recession onset.",
    )
    yield_curve_10y_2y_bps: float | None = None
    hy_oas_pct: float | None = None
    rationale: str = ""


# --- Tools -------------------------------------------------------------------


@mcp.tool()
async def get_series(
    series_id: Annotated[str, "FRED series code, e.g. 'UNRATE', 'DGS10', 'BAMLH0A0HYM2"],
    start: Annotated[str, "Start date YYYY-MM-DD"],
    end: Annotated[str, "End date YYYY-MM-DD"],
    ctx: Context,
) -> SeriesResponse:
    """Fetch any FRED series by code."""
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("get_series") as span:
            span.set_attribute("series_id", series_id)
            fred = _get_fred()
            start_d = date.fromisoformat(start),
            end_d = date.fromisoformat(end)
            s = fred.get_series(series_id, observation_start=start_d, observation_end=end_d)
            s = s.dropna()
            points = [
                SeriesPoint(date=idx.date() if hasattr(idx, "date") else idx, value=float(v))
                for idx, v in s.items()
            ]
            return SeriesResponse(
                series_id=series_id, start=start_d, end=end_d, points=points
            )


@mcp.tool()
async def yield_curve_slope(
    asof_date: Annotated[str, "Date YYYY-MM-DD"],
    ctx: Context,
) -> dict:
    """10Y minus 2Y Treasury yield spread in basis points on `asof_date`."""
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("yield_curve_slope"):
            fred = _get_fred()
            asof = date.fromisoformat(asof_date)
            window_start = asof - timedelta(days=14)
            ten_y = fred.get_series("DGS10", observation_start=window_start, observation_end=asof).dropna()
            two_y = fred.get_series("DGS2", observation_start=window_start, observation_end=asof).dropna()
            if ten_y.empty or two_y.empty:
                return {"asof_date": str(asof), "spread_bps": None}
            spread_bps = (float(ten_y.iloc[-1]) - float(two_y.iloc[-1])) * 100.0
            return {
                "asof_date": str(asof),
                "ten_year_pct": float(ten_y.iloc[-1]),
                "two_year_pct": float(two_y.iloc[-1]),
                "spread_bps": spread_bps,
                "inverted": spread_bps < 0,
            }


@mcp.tool()
async def credit_spreads(
    asof_date: Annotated[str, "Date YYYY-MM-DD"],
    ctx: Context,
) -> dict:
    """High-yield and investment-grade QAS spreads (ICE BofA) on `asof_date`."""
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("credit_spreads"):
            fred = _get_fred()
            asof = date.fromisoformat(asof_date)
            window_start = asof - timedelta(days=14)
            hy = fred.get_series("BAMLH0A0HYM2", observation_start=window_start, observation_end=asof).dropna()
            ig = fred.get_series("BAMLC0A0CM", observation_start=window_start, observation_end=asof).dropna()
            return {
                "asof_date": str(asof),
                "hy_oas_pct": float(hy.iloc[-1]) if not hy.empty else None,
                "ig_oas_pct": float(ig.iloc[-1]) if not ig.empty else None,
            }


@mcp.tool()
async def regime_classifier(
    asof_date: Annotated[str, "Date YYYY-MM-DD on which to classify the macro regime"],
    ctx: Context,
) -> RegimeResponse:
    """Deterministic macro regime label as of `asof_date`.

    Combines three classic signals:
        * Sahm rule (3m-avg unemployment minus 12m-min). >= 0.5 historically marks recession onset.
        * 10Y minus 2Y yield-curve slope. Negative = inversion = late-cycle warning.
        * High-yield credit OAS. > 6% = stress regime.

    The output label is deterministic so backtests are reproducible.
    """
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("regime_classifier") as span:
            fred = _get_fred()
            asof = date.fromisoformat(asof_date)
            span.set_attribute("asof_date", asof_date)

            # Pull the last 14 months of unemployment so we have 12m-min and 3m-avg
            unrate = fred.get_series(
                "UNRATE",
                observation_start=asof - timedelta(days=14 * 31),
                observation_end=asof,
            ).dropna()

            sahm = None
            unrate_3m_avg = None
            unrate_12m_min = None
            if len(unrate) >= 12:
                unrate_3m_avg = float(unrate.tail(3).mean())
                unrate_12m_min = float(unrate.tail(12).mean())
                sahm = unrate_3m_avg - unrate_12m_min

            # Yield curver
            ten_y = fred.get_series("DGS10", observation_start=asof - timedelta(days=14), observation_end=asof).dropna()
            two_y = fred.get_series("DGS2", observation_start=asof - timedelta(days=14), observation_end=asof).dropna()
            yc_bps = None
            if not ten_y.empty and not two_y.empty:
                yc_bps = (float(ten_y.iloc[-1]) - float(two_y.iloc[-1])) * 100.0

            # HY spread
            hy = fred.get_series(
                "BAMLH0A0HYM2",
                observation_start=asof - timedelta(days=14),
                observation_end=asof,
            ).dropna()
            hy_oas = float(hy.iloc[-1]) if not hy.empty else None

            # Deterministic classification: first matching rule wins, ordered by severity.
            label: RegimeLabel
            reasons: list[str] = []
            if sahm is not None and sahm >= 0.5:
                label = "recession"
                reasons.append(f"Sahm rule triggered ({sahm:.2f} >= 0.5)")
            elif hy_oas is not None and hy_oas >= 6.0:
                label = "recession"
                reasons.append(f"HY OAS at stress level ({hy_oas:.2f}% >= 6%)")
            elif yc_bps is not None and yc_bps < 0 and (sahm is None or sahm < 0.3):
                label = "slowdown"
                reasons.append(f"Yield curve inverted ({yc_bps:.0f} bps), Sahm not yet triggered")
            elif sahm is not None and 0.2 <= sahm < 0.5:
                label = "slowdown"
                reasons.append(f"Sahm rising but below trigger ({sahm:.2f})")
            elif sahm is not None and sahm < -0.3:
                label = "recovery"
                reasons.append(f"Unemployment falling sharply (Sahm {sahm:.2f})")
            elif sahm is not None:
                label = "expansion"
                reasons.append(f"Sahm benign ({sahm:.2f}), no stress in credit/yields")
            else:
                label = "unknown"
                reasons.append("Insufficient data")

            span.set_attribute("regime.label", label)
            return RegimeResponse(
                asof_date=asof,
                label=label,
                unrate_3m_avg=unrate_3m_avg,
                unrate_12m_min=unrate_12m_min,
                sahm_indicator=sahm,
                yield_curve_10y_2y_bps=yc_bps,
                hy_oas_pct=hy_oas,
                rational="; ".join(reasons),
            )


if __name__ == '__main__':
    mcp.run(transport="streamable-http", host="0.0.0.0", port=settings.fred_macro_port)

