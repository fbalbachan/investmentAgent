"""Tests for config loading and env resolution in agent.py.

These exercise the pure logic only — no Anthropic API calls and no MCP
subprocesses are launched.
"""

import json

import pytest

import agent


# --------------------------------------------------------------------------- #
# _resolve_env
# --------------------------------------------------------------------------- #

def test_resolve_env_replaces_placeholder(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret123")
    assert agent._resolve_env("${MY_TOKEN}") == "secret123"


def test_resolve_env_missing_var_becomes_empty(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    assert agent._resolve_env("${NOPE}") == ""


def test_resolve_env_embedded_in_string(monkeypatch):
    monkeypatch.setenv("HOST", "example.com")
    assert agent._resolve_env("https://${HOST}/api") == "https://example.com/api"


def test_resolve_env_recurses_into_lists_and_dicts(monkeypatch):
    monkeypatch.setenv("KEY", "abc")
    value = {
        "args": ["--token", "${KEY}"],
        "env": {"NESTED": "${KEY}"},
        "count": 3,          # non-string values pass through untouched
    }
    assert agent._resolve_env(value) == {
        "args": ["--token", "abc"],
        "env": {"NESTED": "abc"},
        "count": 3,
    }


# --------------------------------------------------------------------------- #
# load_mcp_config
# --------------------------------------------------------------------------- #

def _write_config(tmp_path, data):
    path = tmp_path / "mcp_config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_mcp_config_skips_underscore_keys(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, {
        "yahoo_finance": {"command": "uvx", "transport": "stdio"},
        "_disabled": {"command": "nope"},
    })
    monkeypatch.setattr(agent, "CONFIG_PATH", cfg)

    result = agent.load_mcp_config()
    assert list(result.keys()) == ["yahoo_finance"]
    assert "_disabled" not in result


def test_load_mcp_config_resolves_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CHE_API_KEY", "live-key")
    cfg = _write_config(tmp_path, {
        "che": {"command": "npx", "env": {"CHE_API_KEY": "${CHE_API_KEY}"}},
    })
    monkeypatch.setattr(agent, "CONFIG_PATH", cfg)

    result = agent.load_mcp_config()
    assert result["che"]["env"]["CHE_API_KEY"] == "live-key"


def test_load_mcp_config_raises_when_all_disabled(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, {"_only_disabled": {"command": "nope"}})
    monkeypatch.setattr(agent, "CONFIG_PATH", cfg)

    with pytest.raises(RuntimeError, match="No enabled MCP servers"):
        agent.load_mcp_config()


def test_real_config_enables_yahoo_finance_only():
    """The shipped mcp_config.json should expose exactly yahoo_finance today."""
    result = agent.load_mcp_config()
    assert "yahoo_finance" in result
    assert all(not name.startswith("_") for name in result)


def test_default_model_is_opus():
    assert agent.MODEL == "claude-opus-4-8"


# --------------------------------------------------------------------------- #
# StateGraph: routing + wiring
# --------------------------------------------------------------------------- #

from langchain_core.messages import AIMessage
from langgraph.graph import END


def test_should_continue_routes_to_tools_when_tool_calls_present():
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "get_price", "args": {"ticker": "AAPL"}, "id": "1"}],
    )
    assert agent._should_continue({"messages": [ai]}) == "tools"


def test_should_continue_ends_when_no_tool_calls():
    ai = AIMessage(content="Here is my final analysis.")
    assert agent._should_continue({"messages": [ai]}) == END


def test_build_agent_graph_wires_expected_nodes(monkeypatch):
    # A dummy key so ChatAnthropic can construct; no network call is made here.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")

    from langchain_core.tools import tool

    @tool
    def get_price(ticker: str) -> str:
        """Return a fake price for a ticker."""
        return f"{ticker}: 100"

    graph = agent.build_agent_graph([get_price])
    nodes = graph.get_graph().nodes
    assert "agent" in nodes
    assert "tools" in nodes
