"""CLI entry point for the investment-analysis agent.

Usage:
    python main.py "Compare AAPL and MSFT as long-term holds"   # one-shot
    echo "Is NVDA overvalued right now?" | python main.py        # one-shot (piped)
    python main.py            # no args -> interactive session with memory
"""

import asyncio
import os
import sys

import agent


def refresh_windows_path() -> None:
    """Rebuild PATH from the registry (Windows only).

    MCP servers launch as subprocesses (e.g. ``uvx``). If a tool was installed
    after the current shell opened, that shell's PATH is stale and the spawn
    fails with a cryptic ``WinError 2``. Re-reading the machine + user PATH from
    the registry makes freshly-installed tools discoverable without reopening
    the terminal.
    """
    if sys.platform != "win32":
        return
    import winreg

    values = []
    for root, sub in (
        (winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, "Environment"),
    ):
        try:
            with winreg.OpenKey(root, sub) as key:
                val, _ = winreg.QueryValueEx(key, "Path")
                if val:
                    values.append(os.path.expandvars(val))
        except OSError:
            pass

    if values:
        merged = os.pathsep.join(values)
        current = os.environ.get("PATH", "")
        # De-duplicate while preserving order (registry entries first).
        seen, ordered = set(), []
        for part in (merged + os.pathsep + current).split(os.pathsep):
            key = part.lower()
            if part and key not in seen:
                seen.add(key)
                ordered.append(part)
        os.environ["PATH"] = os.pathsep.join(ordered)


EXIT_WORDS = {"exit", "quit", ":q"}


def has_inline_query() -> bool:
    """True when a query came from CLI args or a stdin pipe (one-shot mode)."""
    return len(sys.argv) > 1 or not sys.stdin.isatty()


def read_query() -> str:
    """Get the one-shot query from CLI args or a stdin pipe."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return sys.stdin.read().strip()


async def _run_once() -> int:
    query = read_query()
    if not query:
        print("No query provided.", file=sys.stderr)
        return 1
    answer = await agent.analyze_investments(query)
    print(answer)
    return 0


async def _run_session() -> int:
    """Interactive multi-turn session that remembers earlier questions."""
    print("Investment agent — interactive session with memory.")
    print("Ask follow-ups; earlier questions stay in context.")
    print("Type 'exit' (or Ctrl-C) to quit.\n")

    print("Connecting to data source...", flush=True)
    graph = await agent.create_session()

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in EXIT_WORDS:
            break
        try:
            answer = await agent.ask(graph, query)
        except Exception as exc:  # keep the session alive on a single failure
            print(f"\n[error] {exc}", file=sys.stderr)
            continue
        print(f"\nAgent: {answer}")

    print("Session ended.")
    return 0


async def _run() -> int:
    if has_inline_query():
        return await _run_once()
    return await _run_session()


def main() -> None:
    # Claude's answers contain em-dashes, emoji, etc. The Windows console
    # defaults to cp1252 and would mangle them, so force UTF-8 output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Make freshly-installed tools (e.g. uvx) discoverable in a stale shell.
    refresh_windows_path()

    # MCP servers launch as stdio subprocesses; the Proactor loop on Windows
    # handles subprocess pipes correctly (it is the default on 3.8+, set
    # explicitly here to be safe across environments).
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
