"""CLI entry point for the investment-analysis agent.

Usage:
    python main.py "Compare AAPL and MSFT as long-term holds"
    echo "Is NVDA overvalued right now?" | python main.py
    python main.py            # then type your question at the prompt
"""

import asyncio
import sys

from agent import analyze_investments


def read_query() -> str:
    """Get the query from CLI args, stdin pipe, or an interactive prompt."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return input("Investment question: ").strip()


async def _run() -> int:
    query = read_query()
    if not query:
        print("No query provided.", file=sys.stderr)
        return 1
    answer = await analyze_investments(query)
    print(answer)
    return 0


def main() -> None:
    # Claude's answers contain em-dashes, emoji, etc. The Windows console
    # defaults to cp1252 and would mangle them, so force UTF-8 output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # MCP servers launch as stdio subprocesses; the Proactor loop on Windows
    # handles subprocess pipes correctly (it is the default on 3.8+, set
    # explicitly here to be safe across environments).
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
