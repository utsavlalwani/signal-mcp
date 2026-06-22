"""SEC EDGAR Fundamentals MCP Server.

Point-in-time (PIT) fundamentals -- every tool that returns the financial data takes an `asof_date` parameter, and only data that was *publicly filed before that date* is returned. This eliminates look-ahead bias at the MCP boundary, which is the single biggest source of fake alpha in personal-project backtests.

Built on `edgartools` (the modern Python SEC client).

Port: 8002
"""
from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from edgar import Company, set_identity
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

from agent.config import settings
from agent.tracing import init_tracing, get_tracer, continue_trace_from_meta
from servers._auth import build_verifier


init_tracing("edgar_fundamentals")
tracer = get_tracer(__name__)

# SEC requires that all clients identify themselves. Set this once at startup.
set_identity(settings.edgar_user_agent)


mcp = FastMCP(
    name="edgar-fundamentals",
    instructions=(
        "SEC EDGAR fundamentals tools. EVERY tool takes an `asof_date` to "
        "enforce point-in-time correctness -- only data that was publicly "
        "filed BEFORE asof_date will be returned. This is non-negotiable; "
        "it prevents look-ahead bias in backtests."
    ),
    auth=build_verifier(),
)


# --- Schemas -------------------------------------------------------


class FundamentalsSnapshot(BaseModel):
    """A point-in-time snapshot of a company's most recently-filed fundamentals."""
    ticker: str
    asof_date: str
    most_recent_filing_date: date
    period_end_date: date
    form_type: str
    revenue_ttm: float | None = None
    net_income_ttm: float | None = None
    total_assets: float | None = None
    total_equity: float | None = None
    operating_cash_flow_ttm: float | None = None
    shares_outstanding: float | None = None


class FilingRecord(BaseModel):
    ticker: str
    form_type: str
    filing_date: date
    period_end_date: date | None = None
    accession_number: str
    url: str


class XbrlPoint(BaseModel):
    period_end: date
    filed_date: date
    value: float


class InsiderTxn(BaseModel):
    filing_date: date
    transaction_date: date | None = None
    reporter_name: str
    transaction_code: str
    shares: float | None = None
    price: float | None = None


# --- Helpers ---------------------------------------------


def _get_company_or_raise(ticker: str) -> Company:
    try:
        c = Company(ticker)
        if c is None or not getattr(c, "cik", None):
            raise ValueError(f"Ticker '{ticker}' not found in EDGAR")
        return c
    except Exception as e:
        raise ValueError(f"Could not resolve ticker '{ticker}' on EDGAR: {e}")


# --- Tools ------------------------------------------------------------


@mcp.tool()
async def pit_fundamentals(
    ticker: Annotated[str, "Ticker symbol, e.g. AAPL"],
    asof_date: Annotated[str, "Point-in-time date YYYY-MM-DD. Only filings BEFORE this date will be considered."],
    ctx: Context,
) -> FundamentalsSnapshot:
    """Most-recently-filed fundamentals strictly before `asof_date`. Filings on or after that date are excluded. This is the only honest wat to use fundamentals in a walk-forward backtest.
    """
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("pit_fundamentals") as span:
            span.set_attribute("ticker", ticker)
            span.set_attribute("asof_date", asof_date)

            asof = date.fromisoformat(asof_date)
            company = _get_company_or_raise(ticker)

            # Get periodic filings (10-K, 10-Q), keep only these filed strictly before asof
            try:
                filings = company.get_filings(form=["10-K", "10-Q"])
            except Exception as e:
                raise RuntimeError(f"EDGAR filings fetch failed for {ticker}: {e}")

            # edgartools returns a Filings object; convert to a date-filtered list
            candidate = None
            for f in filings:
                filed_d = f.filing_date if isinstance(f.filing_date, date) else date.fromisoformat(str(f.filing_date))
                if filed_d < asof:
                    candidate = f
                    break   # filings are returned in reverse-chronological order

            if candidate is None:
                raise ValueError(
                    f"No 10-K/10-Q filings for {ticker} available before {asof_date}"
                )

            filed_d = (
                candidate.filing_date
                if isinstance(candidate.filing_date, date)
                else date.fromisoformat(str(candidate.filing_date))
            )
            period_end = (
                candidate.period_of_report
                if isinstance(candidate.period_of_report, date)
                else date.fromisoformat(str(candidate.period_of_report))
            )

            # Pull the financial facts -- edgartools exposes them via .financials
            # We deliberately keep this defensive; some smaller tickers have sparse XBRL.
            snapshot = FundamentalsSnapshot(
                ticker=ticker,
                asof_date=asof,
                most_recent_filing_date=filed_d,
                period_end_date=period_end,
                form_type=candidate.form,
            )

            try:
                financials = candidate.financials
                if financials is not None:
                    income = financials.income_statement().to_dataframe() if hasattr(financials, "income-statement") else None
                    balance = financials.balance_sheet().to_dataframe() if hasattr(financials, "balance_sheet") else None
                    cashflow = financials.cashflow_statement().to_dataframe() if hasattr(financials, "cashflow-statement") else None

                    def _first_val(df, candidates):
                        if df is None or df.empty:
                            return None
                        for c in candidates:
                            if c in df.index:
                                try:
                                    return float(df.iloc[df.index.get_loc(c)].iloc[0])
                                except Exception:
                                    continue

                        return None

                    snapshot.revenue_ttm = _first_val(income, ["Revenues", "Revenue", "SalesRevenueNet"])
                    snapshot.net_income_ttm = _first_val(income, ["NetIncomeLoss", "NetIncome"])
                    snapshot.operating_cash_flow_ttm = _first_val(cashflow, ["NetCashProvidedByUsedInOperatingActivities"])
                    snapshot.total_assets = _first_val(balance, ["Assets"])
                    snapshot.total_equity = _first_val(balance, ["StockholdersEquity"])
                    snapshot.shares_outstanding = _first_val(balance, ["CommonStockSharesOutstanding"])

            except Exception as e:
                span.record_exception(e)
                # Don't fail the call; return what we have. The agent gets a useful snapshot with metadata even if some XBRL concepts are missing.

            return snapshot


@mcp.tool()
async def get_filings(
    ticker: Annotated[str, "Ticker symbol"],
    asof_date: Annotated[str, "Only filings before this date will be returned (YYYY-MM-DD)"],
    form_types: Annotated[list[str], "Form types, e.g. ['10-K', '10-Q', '8-K']"] = ["10-K, 10-Q"],
    limit: Annotated[int, "Max records returned"] = 10,
    ctx: Context = None,
) -> list[FilingRecord]:
    """Recent filings for a ticker, point-in-time filtered."""
    with continue_trace_from_meta(getattr(ctx, "meta", None)) if ctx else None:
        with tracer.start_as_current_span("get_filings"):
            asof = date.fromisoformat(asof_date)
            company = _get_company_or_raise(ticker)

            try:
                filings = company.get_filings(form=form_types)
            except Exception as e:
                raise RuntimeError(f"EDGAR filings fetch failed: {e}")

            out: list[FilingRecord] = []
            for f in filings:
                if len(out) >= limit:
                    break
                filed_d = f.filing_date if isinstance(f.filing_date, date) else date.fromisoformat(str(f.filing_date))
                if filed_d >= asof:
                    continue
                period_end = None
                if f.period_of_report:
                    period_end = (
                        f.period_of_report if isinstance(f.period_of_report, date) else date.fromisoformat(str(f.period_of_report))
                    )
                out.append(FilingRecord(
                    ticker=ticker,
                    form_type=f.form,
                    filing_date=filed_d,
                    period_end_date=period_end,
                    accession_number=f.accession_no,
                    url=f.filing_url if hasattr(f, "filing_url") else "",
                ))
            return out


@mcp.tool()
async def xbrl_concept(
    ticker: Annotated[str, "Ticker symbol"],
    concept: Annotated[str, "XBRL concept name, e.g. 'Revenues', 'NetIncomeLoss', 'Assets'"],
    asof_date: Annotated[str, "Only data filed before this date is returned (YYYY-MM-DD)"],
    ctx: Context,
) -> list[XbrlPoint]:
    """Time series of a single XBRL concept, point-in-time filtered.

    Each point includes both `period_end` (what the value is about) and `filed_date` (when the value first became publicly known). Filter your backtests on `filed_date`, not `period_end`.
    """
    with continue_trace_from_meta(getattr(ctx, "meta", None)):
        with tracer.start_as_current_span("xbrl_concept") as span:
            span.set_attribute("ticker", ticker)
            span.set_attribute("concept", concept)

            asof = date.fromisoformat(asof_date)
            company = _get_company_or_raise(ticker)

            # edgartools exposes a .facts() interface that gives us the full XBRL bag
            try:
                facts = company.get_facts()
            except Exception as e:
                raise RuntimeError(f"EDGAR XBRL facts fetch failed: {e}")

            points: list[XbrlPoint] = []

            try:
                concept_data = facts.get_fact(concept) if hasattr(facts, "get_fact") else None
                if concept_data is not None:
                    for row in concept_data.itertuples() if hasattr(concept_data, "itertuples") else []:
                        filed_d = getattr(row, "filed", None)
                        period_d = getattr(row, "end", None)
                        if filed_d is None or period_d is None:
                            continue
                        filed_d = filed_d if isinstance(filed_d, date) else date.fromisoformat(str(filed_d))
                        if filed_d >= asof:
                            continue
                        period_d = period_d if isinstance(period_d, date) else date.fromisoformat(str(period_d))
                        points.append(XbrlPoint(
                            period_d=period_d,
                            filed_date=filed_d,
                            value=float(getattr(row, "val", 0.0)),
                        ))
            except Exception as e:
                span.record_exception(e)

            return sorted(points, key=lambda p: p.period_end)


@mcp.tool()
async def insider_transactions(
    ticker: Annotated[str, "Ticker symbol"],
    asof_date: Annotated[str, "Only filings before this date (YYYY-MM-DD)"],
    limit: Annotated[int, "Max records"] = 20,
    ctx: Context = None,
) -> list[InsiderTxn]:
    """Form 4 insider transaction filings before asof_date."""
    with continue_trace_from_meta(getattr(ctx, "meta", None) if ctx else None):
        with tracer.start_as_current_span("insider_transactions"):
            asof = date.fromisoformat(asof_date)
            company = _get_company_or_raise(ticker)

            try:
                filings = company.get_filings(form="4")
            except Exception as e:
                raise RuntimeError(f"EDGAR Form 4 fetch failed: {e}")

            out: list[InsiderTxn] = []
            for f in filings:
                if len(out) >= limit:
                    break
                filed_d = f.filing_date if isinstance(f.filing_date, date) else date.fromisoformat(str(f.filing_date))
                if filed_d >= asof:
                    continue

                # Minimal record -- full Form 4 parsing is non-trivial.
                # For a richer impl, use edgartools' Form4 parser directly.
                out.append(InsiderTxn(
                    filing_date=filed_d,
                    reporter_name=getattr(f, "reporter", "") or "",
                    transaction_code="",
                ))
            return out


if __name__ == '__main__':
    mcp.run(transport="streamable-http", host="0.0.0.0", port=settings.edgar_fundamentals_port)
