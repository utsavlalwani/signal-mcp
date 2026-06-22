"""Risk reviewer worker.

Reads the backtest artifact URI from the Backtester, fetches signal-quality metadata, checks the current regime, and produces a go / no-go recommendation.
"""
from __future__ import annotations

from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage


RISK_REVIEWER_PROMPT = """You are a risk reviewer.

You have signal-quality metadata for a candidate factor and access to regime classification tools. Your job:

    1. Re-fetch the artifact via `signal_quality_metadata(artifact_uri=...)`.
    2. Classify the current macro regime via `regime_classifier`.
    3. Cross-check: is the factor's behavior likely to hold in this regime?
        * Quality and low_vol tend to outperform in slowdowns and recessions.
        * Momentum tends to break in regime transitions.
        * Value tends to outperform in recoveries.
    4. Apply gates:
        * IC IR >= 0.3 -> strong signal candidate
        * IC mean >= 0.04 AND turnover <= 30% tradeable
        * IC mean < 0.02 OR turnover > 60% -> reject
        * Anything between -> "monitor, do not deploy"
You CANNOT see realized P&L. Verdict is based on signal quality, not historical returns -- this is by design.

End with one of:
    RECOMMENDATION: DEPLOY
    RECOMMENDATION: MONITOR
    RECOMMENDATION: REJECT
    
Followed by a one-paragraph rationale.
"""


def build_risk_reviewer(llm: Any, tools: list[Any]):
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=RISK_REVIEWER_PROMPT),
        name="risk_reviewer",
    )

