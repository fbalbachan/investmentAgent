# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`Agent_DEMO` is an **investment-analysis agent** built on **LangGraph**. Claude
drives a ReAct loop whose tools come from one or more **MCP servers** declared in
`mcp_config.json`. The agent compares investment options using live financial data.

Data source today is the **Yahoo Finance MCP** server. **CHE MCP** (an Argentine
data gateway, `@artificio/che-mcp`) is not yet released ("Phase 3 — Coming");
when it launches, switching to it is a **config-only change** in `mcp_config.json`
— no Python code changes required.

## Environment

- **Python 3.11+ required.** `langchain-mcp-adapters` (via the `mcp` SDK) needs
  3.10+, and the Yahoo Finance MCP server needs 3.11+. The original `.venv` was
  Python 3.9 and **must be recreated** with 3.11.
- `uvx` (from [uv](https://github.com/astral-sh/uv)) is used to launch the Yahoo
  Finance MCP server as a stdio subprocess. Install uv if not present.
- IDE: PyCharm (`.idea/` present)

## Setup

1. Recreate the virtual environment with Python 3.11:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   Or in bash:

   ```bash
   source .venv/Scripts/activate
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Configure secrets — copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## Run

```powershell
python main.py "Compare AAPL and MSFT as long-term holds"
```

Also accepts a piped query (`echo "..." | python main.py`) or an interactive
prompt (`python main.py` with no args).

## Architecture

- `agent.py` — builds the LangGraph ReAct agent. `load_mcp_config()` reads
  `mcp_config.json`, skips disabled servers (keys starting with `_`), and
  resolves `${VAR}` env placeholders. `analyze_investments(query)` is the async
  entry point. Model defaults to `claude-opus-4-8` (override via `AGENT_MODEL`).
- `main.py` — CLI wrapper; sets the Windows Proactor event loop and calls
  `asyncio.run(...)`.
- `mcp_config.json` — MCP server definitions. `yahoo_finance` is active;
  `_che_mcp_when_released` is a disabled placeholder — remove its leading
  underscore to enable it once CHE MCP ships.
- `.env` / `.env.example` — secrets (`ANTHROPIC_API_KEY`, later CHE creds).

## Swapping in CHE MCP later

1. In `mcp_config.json`, rename `_che_mcp_when_released` → `che_mcp` (drop the `_`).
2. Remove or disable (`_`-prefix) the `yahoo_finance` entry.
3. Add `CHE_API_KEY` / `CHE_JWT` to `.env`.
   Docs: https://github.com/Albano-schz/che-mcp-docs
