"""CLI entry point for the investment-analysis agent.

Usage:
    python main.py "Compare AAPL and MSFT as long-term holds"
    echo "Is NVDA overvalued right now?" | python main.py
    python main.py            # then type your question at the prompt
"""

import asyncio
import os
import sys

from agent import analyze_investments


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
