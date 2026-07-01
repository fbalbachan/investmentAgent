"""Tests for query parsing in main.py (arg / stdin / interactive precedence)."""

import io

import main


class _FakeStdin(io.StringIO):
    """StringIO with a controllable isatty()."""

    def __init__(self, text="", tty=False):
        super().__init__(text)
        self._tty = tty

    def isatty(self):
        return self._tty


def test_read_query_from_args(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "Compare", "AAPL", "and", "MSFT"])
    assert main.read_query() == "Compare AAPL and MSFT"


def test_read_query_from_stdin_pipe(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setattr("sys.stdin", _FakeStdin("Is NVDA overvalued?\n", tty=False))
    assert main.read_query() == "Is NVDA overvalued?"


def test_read_query_interactive_prompt(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setattr("sys.stdin", _FakeStdin("", tty=True))
    monkeypatch.setattr("builtins.input", lambda *a: "  typed question  ")
    assert main.read_query() == "typed question"


def test_args_take_precedence_over_stdin(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "from-args"])
    monkeypatch.setattr("sys.stdin", _FakeStdin("from-stdin", tty=False))
    assert main.read_query() == "from-args"
