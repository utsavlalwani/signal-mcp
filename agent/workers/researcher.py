"""Researcher worker.

Reads the user's researcher question, generates one or more factor hypotheses, and prepares a brief grounded in primary sources (fundamentals, macro).
Outputs a structured `ResearchBrief` for the backtester to consume.
"""
from __future__ import annotations

from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage


RESEARCHER_PROMPT = """You are a quant researcher analyst. Your job is to take a research question in plain English and turn it into a concrete, testable factor hypothesis.

You have these tools available:
    * Market data (OHLCV, universe, corporate actions)
    * SEC EDGAR fundamentals (point-in-time only -- always pass asof_date)
    * FRED macro series + a deterministic regime classifier
    * `list_factors` to see what's pre-built
    
Process:
    1. Identify the underlying economic intuition (why might this signal work?).
    2. Check macro context with the regime classifier -- does this idea fit the current regime?
    3. Pick ONE primary factor from `list_factors` (or propose a composite).
    4. Produce a brief: hypothesis, factor name, universe, date window. Be concise.
    
Rules:
    * NEVER ask for fundamentals data without an asof_date.
    * Cite the specific toll calls you made.
    * If you can't find data, say so plainly.
    
When does, return a short summary ending with the line:
    FACTOR_TO_TEST: <factor_name?
"""


def build_researcher(llm: Any, tools: list[Any]):
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=RESEARCHER_PROMPT),
        name="researcher",
    )

