# Signal MCP — agentic market-research workbench

Four MCP servers (market data, SEC fundamentals, FRED macro, a quant engine) behind a LangGraph supervisor agent that runs systematic-research workflows from natural language: signal generation, walk-forward backtesting, portfolio optimization. The eval harness is *agent-blind*: realized P&L and Sharpe are computed and stored, but the agent only ever sees signal-quality metadata (IC, turnover, decile spread), so it can't optimize for its own scorecard. Fundamentals tools take an `asof_date` enforced at the server boundary, which blocks look-ahead.

Tracks the MCP 2025-11-25 spec, Streamable HTTP transport.

## Architecture

```
LangGraph supervisor (Groq llama-3.3)
  ├─ researcher · backtester · risk-reviewer   (worker nodes)
  └─ MultiServerMCPClient
        ├─ market-data        :8001   yfinance
        ├─ edgar-fundamentals :8002   edgartools (SEC), asof_date-gated
        ├─ fred-macro         :8003   FRED
        └─ quant-engine       :8004   VectorBT walk-forward, PyPortfolioOpt
Postgres — LangGraph checkpointing | Langfuse — per-tool latency/cost
```

Supervisor routes, workers specialize (hypothesize → backtest → review). State is checkpointed to Postgres, so any session replays from any step.

## Setup

Python 3.11/3.12, Docker Desktop. Free keys: FRED, Groq, Langfuse.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d                 # Postgres + Langfuse
cp .env.example .env                 # add FRED_API_KEY, GROQ_API_KEY, LANGFUSE_*
```

## Run

Start the four servers (separate terminals), ports 8001–8004:

```bash
python -m servers.market_data.server
python -m servers.edgar_fundamentals.server
python -m servers.fred_macro.server
python -m servers.quant_engine.server
```

Health check at `http://localhost:8001/health`. Then the demo:

```bash
python -m scripts.run_demo
```

Runs a sample session ("quality+momentum on demo-50, walk-forward 2018–2023, is the signal worth pursuing?"), streams the LangGraph state to stdout, and lands a trace in Langfuse.

## Structure

```
servers/{market_data,edgar_fundamentals,fred_macro,quant_engine}/server.py
servers/quant_engine/{factor_library,backtest,optimizer}.py
agent/{supervisor,client,tracing}.py · agent/workers/* · agent/eval/agent_blind_harness.py
scripts/{run_demo,seed_universe}.py
tests/test_smoke.py
```