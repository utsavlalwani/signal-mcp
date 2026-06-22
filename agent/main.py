"""Agent entrypoint

Wires every component together:
    * LLM (Groq or Anthropic)
    * MultiServerMCPClient pointing at all 4 servers
    * Three workers (researcher, backtester, risk_reviewer)
    * LangGraph supervisor with Postgres checkpointing
    * OTel tracing exporting to Langfuse

Call `run_session(question)` to execute one end-to-end research workflow.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent.config import settings
from agent.client import build_mcp_client, fetch_tools_for_worker
from agent.tracing import init_tracing, get_tracer
from agent.supervisor import build_supervisor_graph
from agent.workers.researcher import build_researcher
from agent.workers.backtester import build_backtester
from agent.workers.risk_reviewer import build_risk_reviewer


def _build_llm():
    """Construct the LLM client based on env config."""
    if settings.agent_llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.agent_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=2048,
        )
    # default: groq
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.agent_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_tokens=2048,
    )


@asynccontextmanager
async def _postgres_checkpointer():
    """Open an AsyncPostgresSaver connection for the duration of a session."""
    async with AsyncPostgresSaver.from_conn_string(settings.postgres_dsn) as saver:
        await saver.setup()
        yield saver


async def run_session(question: str, session_id: str | None = None) -> dict:
    """Execute one full research session and return the resulting state."""
    init_tracing("agent")
    tracer = get_tracer(__name__)

    session_id = session_id or uuid.uuid4().hex

    llm = _build_llm()
    mcp_client = build_mcp_client()

    # Pull tools per-worker (each worker sees only what it needs)
    researcher_tools = await fetch_tools_for_worker(mcp_client, "researcher")
    backtester_tools = await fetch_tools_for_worker(mcp_client, "backtester")
    risk_tools = await fetch_tools_for_worker(mcp_client, "risk_reviewer")

    researcher = build_researcher(llm, researcher_tools)
    backtester = build_backtester(llm, backtester_tools)
    risk_reviewer = build_risk_reviewer(llm, risk_tools)

    config = {"configurable": {"thread_id": session_id}}

    with tracer.start_as_current_span("research_session") as span:
        span.set_attribute("session_id", session_id)
        span.set_attribute("question", question)

        async with _postgres_checkpointer() as checkpointer:
            graph = build_supervisor_graph(
                researcher=researcher,
                backtester=backtester,
                risk_reviewer=risk_reviewer,
                checkpointer=checkpointer,
            )
            initial_state = {
                "user_question": question,
                "research_brief": "",
                "factor_to_test": "",
                "backtest_summary": "",
                "artifact_uri": "",
                "final_verdict": "",
                "iteration": 0,
                "next_node": "researcher",
                "messages": [],
            }
            final_state = await graph.ainvoke(initial_state, config=config)

    return final_state


if __name__ == '__main__':
    import sys
    q = " ".join(sys.argv[1:]) or (
        "Build me a quality+momentum factor on demo-50, walk-forward 2022-01-01 to 2024-12-31, "
        "and tell me whether the signal is worth pursuing."
    )
    asyncio.run(run_session(q))
