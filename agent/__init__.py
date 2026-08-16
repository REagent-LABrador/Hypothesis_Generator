"""The agent layer: what a model needs to drive this codebase well.

    CLAUDE.md   the system prompt — how to run the generator and how to
                present what it returns. This is the deliverable; the code
                below only makes it callable.
    tools.py    six tools over the core and the adapters, plus their schemas
    server.py   an MCP stdio server exposing them

The split matters. The generator decides *what* is true of a graph; the prompt
decides *how honestly it is reported*, which is the half a model can get wrong
without anything failing. Neither file duplicates the other's job.
"""
