"""The MCP server: that it advertises the tools, and that a failure reads well.

The server is thin on purpose, so there is little to test beyond the seam —
that what ``tools.py`` defines is what a client actually receives, and that a
guard rejecting a hostile path surfaces as a message rather than failing open.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "mcp", reason="the agent extra is optional: pip install -e '.[agent]'"
)

import anyio  # noqa: E402

from agent import server, tools  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def run(call):
    """Drive one coroutine factory to completion.

    anyio arrives with the MCP stack, so the async seam is exercised without
    adding a pytest plugin and a backend fixture to configure.
    """
    return anyio.run(call)


@pytest.fixture
def runs(tmp_path: Path, monkeypatch) -> Path:
    target = tmp_path / "runs"
    monkeypatch.setattr(tools, "RUNS", target)
    return target


def test_the_server_advertises_every_tool():
    listed = run(server.build().list_tools)
    assert {t.name for t in listed} == set(tools.BY_NAME)


def test_the_advertised_schema_is_the_one_derived_from_the_signature():
    """One source of truth: the annotations. Not a hand-kept parallel copy."""
    listed = {t.name: t for t in run(server.build().list_tools)}
    for name, tool in listed.items():
        expected = tools.BY_NAME[name]["input_schema"]
        assert set(tool.input_schema.get("properties", {})) == set(
            expected.get("properties", {})
        )


def test_the_advertised_description_is_the_docstring():
    listed = {t.name: t for t in run(server.build().list_tools)}
    assert "THE ONE hypothesis" in listed["generate_hypothesis"].description
    assert "WITHOUT spending a model call" in listed["preview_candidates"].description


def test_the_server_instructions_point_at_the_prompt():
    """Tools without the prompt are a generator with no guidance on reporting."""
    built = server.build()
    assert "CLAUDE.md" in built.instructions
    assert "get_evidence" in built.instructions


def test_a_real_call_round_trips_as_json(runs: Path):
    async def call():
        return await server.build().call_tool(
            "preview_candidates",
            {"graph": "knowledge-graph.json", "profile": "repurposing"},
        )

    result = run(call)
    payload = json.loads(result.content[0].text)
    assert payload["graph_id"] == "g_demo1"
    assert payload["candidates"]


def test_a_rejected_path_surfaces_its_reason(runs: Path):
    """The guard must not fail open, and the message has to reach the model."""
    from mcp.server.mcpserver.exceptions import ToolError as McpToolError

    async def call():
        return await server.build().call_tool(
            "preview_candidates", {"graph": "../../etc/passwd"}
        )

    with pytest.raises(McpToolError, match="bare filename"):
        run(call)
