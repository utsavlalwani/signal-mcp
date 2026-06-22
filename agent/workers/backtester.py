"""Backtester worker.

Runs a walk-forward backtest of the factor proposed by the Researcher.
Returns ONLY signal-quality metadata (this restriction is enforced serve-side in the quant-engine MCP -- the backtester cannot see realized P&L even if it asks).
"""
from __future__ import annotations
from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage


BACKTESTER_PROMPT = """You are a quant backtester.

Your input is a factor name proposed by the researcher. Your job:

    1. Call `walk_forward_backtest` with sensible defaults:
        * universe_name: "demo-50"
        * start/end: the date range the researcher specified
        * train_window_months: 24
        * test_window_months: 1
        * long_short: True
        * transaction_cost_bps: 10
    2. Reads the response -- note IC, IC information ratio, turnover, and decile spread.
    3. If IC is below 0.02, the factor is probably noise -- say so plainly.
    4. If turnover is above 50% per month, flag it -- the factor will be eaten by costs.
    5. Pass the `artifact_uri` forward so the risk reviewer can re-examine the signal.
    
You CANNOT see realized P&L, Sharpe, or drawdown. The system is designed this way on purpose to prevent metric gaming. Don't try to work around it.

When done, return a 4-line summary:
    FACTOR: <name>
    IC_MEAN: <value>
    TURNOVER: <pct>
    ARTIFACT_URI: <uri>
"""


def build_backtester(llm: Any, tools: list[Any]):
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=BACKTESTER_PROMPT),
        name="backtester",
    )
