"""Market Data MCP Server.

Exposes adjusted-close OHLCV panels, universes, and corporate actions over a unified schema. Provider is configurable (yfinance default; Polygon and Tiingo supported via env var).

Port: 8001 (configurable via MARKET_DATA_PORT)
Transport: Streamable HTTP
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Annotated, Literal

import pandas as pd
import yfinance as yf
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

from agent.config import settings
from agent.tracing import init_tracing, get_tracer, continue_trace_from_meta
from servers._auth import build_verifier


init_tracing("market_data")
tracer = get_tracer(__name__)

mcp = FastMCP(
    name="market-data",
    instructions=(
        "Market data tools. Use `get_universe` to enumerate tickers, then "
        "`get_ohlcv` to pull adjusted-close panels. Dates are ISO YYYY-MM-DD."
    ),
    auth=build_verifier(),
)


# --- Schemas -----------------------------------------------------------


class OhlcvRow(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float = Field(..., description="Split- and dividend-adjusted close" )
    volume: float


class OhlcvPanel(BaseModel):
    ticker: str
    start: date
    end: date
    n_rows: int
    rows: list[OhlcvRow]


class UniverseResponse(BaseModel):
    name: str
    tickers: list[str]
    as_of: date


class CorporateAction(BaseModel):
    date: date
    type: Literal["split", "dividend"]
    value: float


# --- Hard-coded "demo-50" universe ----------------------------------------
# In production, this would come from an index-membership database. For a local portfolio demo, a static list like below is honest and keeps the API calls cheap.

DEMO_50 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "LLY", "ABBV", "UNH", "V", "ХОМ", "MA", "GOOGL", "META", "TSLA", "TMO", "CVX", "WMT", "BAC", "KO", "PG", "JNJ", "HD", "COST", "MRK", "BRK-B", "AVGO", "JPM", "ACN", "ABT", "NFLX", "PEP", "CRM", "ORCL", "ADBE", "MCD", "VZ", "AMD", "INTC", "QCOM", "LIN", "DHR", "WFC", "TXN", "DIS", "INTU", "CSCO", "PFE", "CMCSA", "PM", "BMY", "IBM",
]


def _resolve_universe(name: str) -> list[str]:
    name_lower = name.lower()
    if name_lower in {"demo-50", "demo50", "demo"}:
        return DEMO_50
    if name_lower in {"sp10", "sp-10", "demo-10"}:
        return DEMO_50[:10]
    raise ValueError(
        f"Universe '{name}' not recognized. Available: demo-50, demo-10"
    )


# --- Tools ------------------------------------------------------------------------


@mcp.tool()
async def get_universe(
    name: Annotated[str, "Universe name: demo-50 or demo-10"],
    ctx: Context,
) -> UniverseResponse:
    """List tickers belonging to a named universe."""
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("get_universe") as span:
            tickers = _resolve_universe(name)
            span.set_attribute("universe.name", name)
            span.set_attribute("universe.size", len(tickers))
            return UniverseResponse(name=name, tickers=tickers, as_of=date.today())


@mcp.tool()
async def get_ohlcv(
    ticker: Annotated[str, "Ticker symbol, e.g. AAPL"],
    start: Annotated[str, "Start date, ISO format YYYY-MM-DD"],
    end: Annotated[str, "End date inclusive, ISO format YYYY-MM-DD"],
    ctx: Context,
) -> OhlcvPanel:
    """Adjusted-close OHLCV panel from the configured market data provider."""
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("get_ohlcv") as span:
            span.set_attributes({
                "ticker": ticker,
                "start": start,
                "end": end,
                "provider": settings.market_data_provider,
            })

            start_d = date.fromisoformat(start)
            end_d = date.fromisoformat(end)

            if settings.market_data_provider != "yfinance":
                # Polygon / Tiingo paths would go here. yfinance is the demo default.
                raise NotImplementedError(
                    f"Provider {settings.market_data_provider} not implemented in demo"
                )

            df = yf.download(
                ticker,
                start=start_d,
                end=end_d,
                progress=False,
                auto_adjust=False,
            )

            if df.empty:
                return OhlcvPanel(
                    ticker=ticker, start=start_d, end=end_d, n_rows=0, rows=[]
                )

            # Handle yfinance multi-index columns for single ticker
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            rows = [
                OhlcvRow(
                    date=idx.date(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Close"]),
                    close=float(row["Adj Close"]),
                    volume=float(row["Volume"]),
                )
                for idx, row in df.iterrows()
                if pd.notna(row["Close"])
            ]
            span.set_attribute("rows,returned", len(rows))
            return OhlcvPanel(
                ticker=ticker, start=start_d, end=end_d, n_rows=len(rows), rows=rows
            )


@mcp.tool()
async def get_corporate_actions(
    ticker: Annotated[str, "Ticker symbol"],
    start: Annotated[str, "Start date YYYY-MM-DD"],
    end: Annotated[str, "End date YYYY-MM-DD"],
    ctx: Context,
) -> list[CorporateAction]:
    """Splits and dividends in the date window."""
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        t = yf.Ticker(ticker)
        actions: list[CorporateAction] = []

        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)

        splits = t.splits
        for idx, value in splits.items():
            d = idx.date() if hasattr(idx, "date") else idx
            if start_d <= d <= end_d and value != 1.0:
                actions.append(CorporateAction(date=d, type="split", value=float(value)))

        divs = t.dividends
        for idx, value in divs.items():
            d = idx.date() if hasattr(idx, "date") else idx
            if start_d <= d <= end_d:
                actions.append(CorporateAction(date=d, type="dividend", value=float(value)))

        return sorted(actions, key=lambda a: a.date)


@mcp.tool()
async def get_market_calendar(
    start: Annotated[str, "Start date YYYY-MM-DD"],
    end: Annotated[str, "End date YYYY-MM-DD"],
    ctx: Context,
) -> list[date]:
    """NYSE.NASDAQ trading session dates in the window."""
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("get_market_calendar"):
            # Use SPY as a market proxy -- its trading days == NYSE/NASDAQ calendar
            df = yf.download(
                "SPY",
                start=date.fromisoformat(start),
                end=date.fromisoformat(end) + timedelta(days=1),
                progress=False,
                auto_adjust=False,
            )
            if df.empty:
                return []
            return [idx.date() for idx in df.index]


if __name__ == '__main__':
    mcp.run(transport="streamable-http", host="0.0.0.0", port=settings.market_data_port)

