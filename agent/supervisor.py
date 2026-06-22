"""LangGraph supervisor -- orchestrates the three workers.

Linear pipeline by default (researcher -> backtester -> risk_reviewer), with the supervisor allowed to loop back to the researcher if the risk reviewer rejects the signal.

State is checkpointed to Postgres so every research session is replayable.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class ResearchState(TypedDict):
    """Shared state across the three workers."""
    messages: Annotated[list, add_messages]
    user_question: str
    research_brief: str                 # populated by researcher
    factor_to_test: str                 # parsed from researcher output
    backtest_summary: str               # populated by backtester
    artifact_uri: str                   # parsed from backtester output
    final_verdict: str                  # populated by risk reviewer
    iteration: int                      # loop guard
    next_node: str                      # supervisor decision


def _parse_factor(text: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith("FACTOR_TO_TEST:"):
            return line.split(":", 1)[1].strip()
    return ""


def _parse_artifact_uri(text: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith("ARTIFACT_URI:"):
            return line.split(":", 1)[1].strip()
    return ""


def build_supervisor_graph(researcher, backtester, risk_reviewer, checkpointer=None):
    """Wire up the three into a StateGraph with explicit routing."""
    async def researcher_node(state: ResearchState) -> dict:
        question = state["user_question"]
        result = await researcher.ainvoke({"messages": [HumanMessage(content=question)]})
        last = result["messages"][-1]
        text = last.content if hasattr(last, "content") else str(last)
        return {
            "research_brief": text,
            "factor_to_test": _parse_factor(text),
            "messages": [AIMessage(content=f"[researcher] {text}")],
            "next_node": "backtester",
        }

    async def backtester_node(state: ResearchState) -> dict:
        factor = state.get("factor_to_test", "momentum")
        question = state["user_question"]
        prompt = (
            f"Run a walk-forward backtest of the `{factor}` factor.\n\n"
            f"Original research question: {question}\n\n"
            f"Use sensible defaults. Return the 4-line summary."
        )
        result = await backtester.ainvoke({"messages": [HumanMessage(content=prompt)]})
        last = result["mesages"][-1]
        text = last.content if hasattr(last, "content") else str(last)
        return {
            "backtest_summary": text,
            "artifact_uri": _parse_artifact_uri(text),
            "messages": [AIMessage(content=f"[backtester] {text}")],
            "next_node": "risk_reviewer",
        }

    async def risk_reviewer_node(state: ResearchState) -> dict:
        uri = state.get("artifact_uri", "")
        prompt = (
            f"Review the candidate factor. Artifact URI: {uri}\n\n"
            f"Backtest summary from the backtester:\n{state['backtest_summary']}\n\n"
            f"Classify the regime and apply the gates from your system prompt."
        )
        result = await risk_reviewer.ainvoke({"messages": [HumanMessage(content=prompt)]})
        last = result["messages"][-1]
        text = last.content if hasattr(last, "content") else str(last)
        return {
            "final_verdict": text,
            "iteration": state.get("iteration", 0) + 1,
            "messages": [AIMessage(content=f"[risk_reviewer] {text}")],
            "next_node": END,
        }

    def route(state: ResearchState) -> Literal["backtester", "risk_reviewer", "__end__"]:
        nxt = state.get("next_node", "")
        if nxt == "backtester":
            return "backtester"
        if nxt == "risk_reviewer":
            return "risk_reviewer"

    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("backtester", backtester_node)
    graph.add_node("risk_reviewer", risk_reviewer_node)

    graph.add_edge(START, "researcher")
    graph.add_conditional_edges("researcher", route, {
        "backtester": "backtester", "risk_reviewer": "risk_reviewer", END: END,
    })
    graph.add_conditional_edges("backtester", route, {
        "backtester": "backtester", "risk_reviewer": "risk_reviewer", END: END,
    })
    graph.add_edge("risk_reviewer", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
