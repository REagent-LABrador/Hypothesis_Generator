"""Adapters: everything that turns a hypothesis into something else.

The core (``hyp_gen``) reads a knowledge graph and writes one
``hypothesis.json``. It stops there on purpose. Every other artifact anyone has
wanted from it -- a report to read, a payload for a web UI, an SVG of the walk,
a brief for the valuation model -- is produced here instead, by a program that
reads that file and never touches the graph.

    adapters/report/     hypothesis.json -> report.md (four modes)
    adapters/webui/      hypothesis.json -> cards.json + traces.svg
    adapters/valuation/  hypothesis.json -> *.program.json for the ROI model

Each directory carries its own SCHEMA.md, covering both sides. The shared
rules, and the four helpers that keep a failure looking like a failure in every
rendering, are in ``adapters/common.py``.

The dependency runs one way only: adapters import ``hyp_gen``; ``hyp_gen``
never imports an adapter, and no adapter imports another.
"""
