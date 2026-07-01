"""Tests for query parsing and PATH refresh in main.py."""

import io
import os
import sys

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


def test_args_take_precedence_over_stdin(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "from-args"])
    monkeypatch.setattr("sys.stdin", _FakeStdin("from-stdin", tty=False))
    assert main.read_query() == "from-args"


def test_has_inline_query_true_for_args(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "AAPL"])
    monkeypatch.setattr("sys.stdin", _FakeStdin("", tty=True))
    assert main.has_inline_query() is True


def test_has_inline_query_true_for_pipe(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setattr("sys.stdin", _FakeStdin("piped", tty=False))
    assert main.has_inline_query() is True


def test_has_inline_query_false_for_interactive(monkeypatch):
    """No args + a real terminal -> interactive session (not one-shot)."""
    monkeypatch.setattr("sys.argv", ["main.py"])
    monkeypatch.setattr("sys.stdin", _FakeStdin("", tty=True))
    assert main.has_inline_query() is False


def test_refresh_windows_path_noop_off_windows(monkeypatch):
    """On non-Windows platforms it must do nothing and not touch PATH."""
    monkeypatch.setattr(sys, "platform", "linux")
    before = os.environ.get("PATH", "")
    main.refresh_windows_path()
    assert os.environ.get("PATH", "") == before
