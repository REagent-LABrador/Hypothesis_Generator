"""MCP stdio server exposing the tools in ``tools.py``.

    hypagent                  # after `pip install -e '.[agent]'`
    python -m agent.server

Registered with Claude Code by the repo's ``.mcp.json``; any MCP client can
speak to it the same way. The server is deliberately thin — it registers six
plain Python functions and lets the transport do the rest — so everything worth
testing lives in ``tools.py`` and is tested without a transport in the way.

**When this file earns its place.** An agent with shell access to this machine
does not need it: ``hypgen`` writes the document and the adapters read it, so
the CLI is already the tool surface. This exists for the agent that *cannot*
reach a shell here — a hosted agent in a cloud sandbox, a desktop or web
client, anything driving the API directly. See ``README.md`` beside this file.

``CLAUDE.md`` beside this file is the other half, and the more important one. A
client that loads these tools without that prompt gets a working generator and
no guidance on how to report what it returns honestly, which is exactly the half
a model gets wrong without anything failing.
"""

from __future__ import annotations

from agent import tools

try:
    from mcp.server import MCPServer
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "the MCP server needs the `mcp` package: pip install -e '.[agent]'"
    ) from exc


NAME = "hyp-gen"

INSTRUCTIONS = """\
Turns a literature knowledge graph into one evidence-traceable hypothesis.

Read agent/CLAUDE.md in this repository before using these tools. It is the
operating manual: which profile and craziness to pick, how to read a
verification gate table, and what may never be said about a result. The tools
enforce what the graph contains; the prompt is what keeps the reporting honest.

The short version: preview first (it is free), leave articulate false until the
user wants prose, always call get_evidence before presenting, and never state a
relationship the graph does not contain.\
"""


def build() -> MCPServer:
    """Register every tool in ``tools.HANDLERS``.

    The description each tool advertises is its docstring and the parameter
    docs come from the annotations, so this loop adds no prose of its own —
    there is nowhere for a second, staler copy to live.
    """
    server = MCPServer(name=NAME, instructions=INSTRUCTIONS)
    for handler in tools.HANDLERS:
        server.add_tool(handler, description=tools.describe(handler))
    return server


def main() -> int:
    build().run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
