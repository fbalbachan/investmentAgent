# Agent_DEMO — Investment-Analysis Agent

An investment-analysis agent that compares investment options using **live
financial data**. Claude drives a [LangGraph](https://langchain-ai.github.io/langgraph/)
ReAct loop whose tools come from an **MCP** ([Model Context Protocol](https://modelcontextprotocol.io/))
server. The data source is pluggable and declared entirely in config.

- **Today:** [Yahoo Finance MCP](https://github.com/Alex2Yang97/yahoo-finance-mcp) — stock prices, financial statements, options, analyst recommendations.
- **Later:** [CHE MCP](https://github.com/Albano-schz/che-mcp-docs) (an Argentine data gateway) is not yet released. When it launches, switching to it is a **config-only change** — no Python edits required.

## Stack

The **agentic framework is [LangGraph](https://langchain-ai.github.io/langgraph/)** —
it runs the agent loop. `agent.py` builds an explicit LangGraph `StateGraph` (the
`agent` ↔ `tools` ReAct loop with a conditional edge), compiled with
`builder.compile(...)` and executed via `graph.ainvoke(...)`. LangGraph also
supplies the pieces that make this an *agent* rather than a single call:

- **`StateGraph` + `add_messages`** — the graph and the reducer that accumulates the conversation.
- **`MemorySaver` checkpointer** — multi-turn session memory (see [Usage](#usage)).
- **Conditional edges** — the routing that decides "call another tool" vs. "finish".

Supporting libraries (around the framework, not the framework itself):

- **`langchain-anthropic`** (`ChatAnthropic`) — the LLM binding to Claude (`claude-opus-4-8`).
- **`langchain-mcp-adapters`** (`MultiServerMCPClient`) — turns an MCP server's tools into tools LangGraph can call.

In short: **LangGraph = orchestration / agentic framework**, Claude = the model,
MCP = the pluggable tool/data source.

## How it works

```
        query ──▶ main.py ──▶ agent.analyze_investments()
                                   │
                                   ├─ ChatAnthropic (claude-opus-4-8)
                                   │
                                   └─ MultiServerMCPClient ──▶ MCP server (stdio subprocess)
                                                                  └─ Yahoo Finance tools
```

`agent.py` loads the server list from `mcp_config.json`, exposes each server's
tools to Claude, and runs the ReAct loop as an explicit LangGraph `StateGraph`:

```
START ─▶ agent ──(tool calls?)──▶ tools ─▶ agent ─▶ ... ─▶ END
             └────────(no tool calls)───────────────────▶ END
```

- **`agent` node** — calls Claude (with tools bound) on the running message history.
- **`tools` node** — executes whatever tools Claude requested and appends the results.
- A **conditional edge** loops back to `agent` while Claude keeps requesting
  tools, and routes to `END` once it produces a final answer.

The graph is hand-wired (rather than using `create_react_agent`) so there's room
to add nodes for validation, guardrails, or multi-stage analysis. Tool failures
are caught per-call and fed back to Claude, and an `AGENT_RECURSION_LIMIT`
(default 25) caps the loop.

## Requirements

- **Python 3.11+** — required by the `mcp` SDK and the Yahoo Finance MCP server.
- **[uv](https://github.com/astral-sh/uv)** — provides `uvx`, which launches the
  Yahoo Finance MCP server as a subprocess.
- An **Anthropic API key**.

## Setup

```powershell
# 1. Create a Python 3.11 virtual environment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # bash: source .venv/Scripts/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
Copy-Item .env.example .env
#   then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

If `uv` isn't installed: `winget install --id astral-sh.uv -e` (Windows), or see
the [uv install docs](https://github.com/astral-sh/uv#installation).

## Usage

```powershell
# As a command-line argument
python main.py "Compare AAPL and MSFT as long-term holds"

# Piped from stdin
echo "Is NVDA overvalued right now?" | python main.py

# Interactive session with memory — run with no args
python main.py
```

Running with **no arguments** (in a real terminal) starts an **interactive
session that remembers earlier questions**, so follow-ups resolve against
context:

```
$ python main.py
Investment agent — interactive session with memory.

You: How is MELI doing this year?
Agent: ...

You: How does it compare to Amazon?      # "it" = MELI, remembered
Agent: ...

You: exit
```

Memory is kept in-process (via a LangGraph `MemorySaver` checkpointer) and lasts
for the life of the session — it's cleared when you exit. A single query passed
as an argument or piped in still runs one-shot (no memory).

> The **first run is slower**: `uvx` downloads and builds the Yahoo Finance MCP
> server, then caches it for subsequent runs.

### Example

```
$ python main.py "In one sentence, is NVDA more expensive than AMD by P/E right now?"

No — NVDA is actually much cheaper than AMD on a P/E basis right now, trading at a
trailing P/E of ~30.7 (forward ~15.7) versus AMD's ~192.4 (forward ~44.1).
```

## Configuration

| File | Purpose |
|------|---------|
| `mcp_config.json` | MCP server definitions. Keys starting with `_` are disabled. `${VAR}` values are resolved from the environment. |
| `.env` | Secrets (`ANTHROPIC_API_KEY`, later CHE creds). Never committed. |

**Model** defaults to `claude-opus-4-8`. Override with the `AGENT_MODEL`
environment variable.

## Project layout

| File | Role |
|------|------|
| `agent.py` | Builds the LangGraph `StateGraph` ReAct loop; `load_mcp_config()`, `build_agent_graph()`, `analyze_investments(query)`. |
| `main.py` | CLI wrapper — reads the query, sets UTF-8 output + Windows Proactor loop, runs the agent. |
| `mcp_config.json` | MCP server list (`yahoo_finance` active; CHE MCP placeholder). |
| `requirements.txt` | Python dependencies. |
| `requirements-dev.txt` | Test dependencies (`pytest`), layered on `requirements.txt`. |
| `.env.example` | Template for secrets. |
| `tests/` | Unit tests (`pytest`). See [Running tests](#running-tests). |

## Swapping in CHE MCP later

When CHE MCP is released:

1. In `mcp_config.json`, rename `_che_mcp_when_released` → `che_mcp` (drop the `_`).
2. Remove or disable (prefix with `_`) the `yahoo_finance` entry.
3. Add `CHE_API_KEY` / `CHE_JWT` to `.env`.

No changes to `agent.py` or `main.py` are needed. Docs:
https://github.com/Albano-schz/che-mcp-docs

## Running tests

The test suite covers the config-loading logic (`agent.py`) and CLI query
parsing (`main.py`). The tests are fast and fully offline — **no API key, no
network, and no MCP subprocess** — so they're safe for CI and pre-commit hooks.

```powershell
# Install dev dependencies (pytest), then run the suite
pip install -r requirements-dev.txt
pytest
```

Useful variations:

```powershell
pytest tests/test_agent.py        # a single file
pytest -k resolve_env             # tests matching a keyword
pytest -q                         # quieter output
```

## Notes

- `temperature` is **not** set on the model — it is deprecated (and rejected) on
  Opus 4.6+ models.
- This tool produces **analysis, not personalized financial advice**. Always do
  your own due diligence before making investment decisions.
