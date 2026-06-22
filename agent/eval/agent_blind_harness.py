"""Agent-blind evaluation harness: which backtest metrics the LLM may see vs. which stay withheld.

This module is the conceptual home of the agent-blind eval pattern, even though enforcement happens in `servers/quant_engine/backtest.py` and `servers/quant_engine/server.py`.

Why this matters: an LLM that can observe its own strategy's Sharpe ratio will reverse-engineer the metric -- proposing trades that game the number rather than capture genuine signal. The cure is to make the LLM literally incapable of seeing P&L. We enforce this at the MCP server boundary, not in prompts, because prompts can be jailbroken; tool schemas cannot.

The split:

    Visible to LLM (returned by `walk_forward_backtest`):
        * factor_name
        * n_periods
        * ic_mean, ic_ir
        * turnover_pct_per_month
        * decile_spread_bps_per_month
        * artifact_uri (opaque pointer)

    Hidden from LLM (persisted to disk, viewable in Langfuse by the human):
        * realized_returns_pct (per period)
        * sharpe, sortino
        * max_drawdown_pct
        * cumulative_return_pct
        * full equity curve
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentBlindPolicy:
    """Documents which metrics are LLM-visible vs withheld."""

    visible_to_llm: tuple[str, ...] = (
        "factor_name",
        "n_periods",
        "ic_mean",
        "ic_ir",
        "turnover_pct_per_month",
        "decile_spread_bps_per_month",
        "artifact_uri",
    )

    withheld_from_llm: tuple[str, ...] = (
        "realized_returns_pct",
        "sharpe",
        "sortino",
        "max_drawdown_pct",
        "cumulative",
        "equity_curve",
    )

    def is_visible(self, field: str) -> bool:
        return field in self.visible_to_llm

    def is_withheld(self, field: str) -> bool:
        return field in self.withheld_from_llm


POLICY = AgentBlindPolicy()
