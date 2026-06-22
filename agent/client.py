"""MCP client setup -- talks to all 4 MCP servers from the agent.

Uses `langchain-mcp-adapters` MultiServerMCPClient with the Streamable HTTP transport. Returns a list of LangChain tools ready to bind to the LangGraph workers.
"""
from __future__ import annotations

from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.config import settings
from agent.auth import mint_agent_token


def build_mcp_client() -> MultiServerMCPClient:
    """Construct the MCP client targeting all 4 local servers."""
    token = mint_agent_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def _cfg(name: str) -> dict:
        return {"url": settings.server_urls[name], "transport": "streamable_http", "headers": headers}

    return MultiServerMCPClient({
        "market_data": _cfg("market_data"),
        "edgar_fundamentals": _cfg("edgar_fundamentals"),
        "fred_macro": _cfg("fred_macro"),
        "quant_engine": _cfg("quant_engine"),
    })


async def fetch_all_tools(client: MultiServerMCPClient) -> list[Any]:
    """Pull every tool from every server as LangChain Tool objects."""
    return await client.get_tools()


async def fetch_tools_for_worker(client: MultiServerMCPClient, worker: str) -> list[Any]:
    """Filter tools so each worker sees only the ones it needs.

    The Risk Reviewer DELIBERATELY does NOT receive the agent-blind tools (`signal_quality_metadata` is allowed; raw P&L is not exposed via MCP at all).
    """
    all_tools = await fetch_all_tools(client)

    if worker == "researcher":
        # Sees market data, fundamentals, and macro -- but not the quant engine
        allowed = {
            "get_universe", "get_ohlcv", "get_corporate_actions", "get_market_calendar", "pit_fundamentals", "get_filings", "xbrl_concept", "insider_transactions", "get_series", "regime_classifier", "yield_curve_slope", "credit_spreads",
        }
    elif worker == "backtester":
        # Compute / backtest tools + market data
        allowed = {
            "get_universe", "get_ohlcv", "compute_factor", "factor_ic", "walk_forward_backtest", "compute_signal_decile_spread", "factor_turnover", "signal_quality_metadata",
        }
    elif worker == "risk_reviewer":
        # Only macro context + the metadata view of backtest artifacts
        allowed = {
            "regime_classifier", "yield_curve_slope", "credit_spreads", "signal_quality_metadata", "factor_turnover",
        }
    else:
        return all_tools

    return [t for t in all_tools if t.name in allowed]

