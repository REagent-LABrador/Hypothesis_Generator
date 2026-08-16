"""Bundle → a static SVG of the traces the deterministic half walked.

SVG rather than PNG because it is text: no plotting dependency (this package
ships pydantic and the Anthropic SDK and nothing else), it diffs, and it embeds
in a page or a markdown file unchanged. It is a pure function of the record like
every other view here, so the picture can be regenerated from a saved
``hypotheses.json`` without re-running anything.

**What the colours mean, and where they come from.** There is no flag on an
edge saying "contradicted". A link carries three lists of finding ids -- ``yes``,
``no`` and ``no_effect`` -- and the graph builder derives ``state`` from them. So:

- **red** -- the link carries at least one finding that argues against it
  (``no``) or reports no effect (``no_effect``). ``state`` is usually
  ``disagreed``. The edge is still drawn: a contradicted link is part of the
  trace, and hiding it would be the one edit that makes a chain look stronger
  than it is.
- **blue** -- every finding on the link supports it.
- **grey** -- no verbatim finding at all. Not neutral: an edge nobody has
  evidenced.

Dashes are the second axis, and they are independent of colour: **dashed**
means ``single_source`` (one research group), **solid** means the finding set
spans more than one. A solid red edge and a dashed blue one are very different
situations and the picture has to keep them apart.

**Nodes are deduplicated across traces.** Where two hypotheses cross the same
node -- pirfenidone and nintedanib both reaching myofibroblast differentiation --
the traces visibly converge, which is a fact about the graph worth seeing. The
layout is layered left to right by longest path from any trace's subject, which
is deterministic and needs no solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import escape

from adapters.common import Bundle

# -- palette ---------------------------------------------------------------

SUPPORTED = "#2563eb"       # blue: every finding on this link supports it
CONTRADICTED = "#dc2626"    # red: at least one finding argues against it
UNEVIDENCED = "#9ca3af"     # grey: no verbatim finding at all

_INK = "#111827"
_MUTED = "#6b7280"
_PAPER = "#ffffff"
_NODE_FILL = "#f9fafb"
_NODE_STROKE = "#d1d5db"

# -- layout ----------------------------------------------------------------

_COL_W = 268
_BOX_W = 188
_LINE_H = 15
_ROW_GAP = 34
_PAD = 28
_TITLE_H = 64
_LEGEND_H = 96
_CHAR_W = 6.6          # ~13px system sans; only needs to be close
_MAX_LINES = 3


def _wrap(text: str, width_px: float = _BOX_W - 18) -> list[str]:
    """Greedy wrap on an estimated character width.

    Approximate on purpose: SVG has no measurement API without a renderer, and
    a box that is a few pixels loose costs nothing while a layout dependency
    would cost a lot.
    """
    limit = max(1, int(width_px / _CHAR_W))
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == _MAX_LINES - 1 and len(current) > limit:
            current = current[: limit - 1] + "…"
            break
    if current:
        lines.append(current)
    if len(lines) > _MAX_LINES:
        lines = lines[: _MAX_LINES - 1] + [lines[_MAX_LINES - 1][: limit - 1] + "…"]
    return lines or [""]


@dataclass
class _Node:
    id: str
    name: str
    col: int = 0
    row: int = 0
    bary: float = 0.0
    lines: list[str] = field(default_factory=list)

    @property
    def height(self) -> float:
        return len(self.lines) * _LINE_H + 16

    @property
    def x(self) -> float:
        return _PAD + self.col * _COL_W

    @property
    def y(self) -> float:
        return _PAD + _TITLE_H + self.row * (_MAX_LINES * _LINE_H + 16 + _ROW_GAP)

    @property
    def cx(self) -> float:
        return self.x + _BOX_W / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2


@dataclass
class _Edge:
    src: str
    dst: str
    link: str
    how: str
    colour: str
    dashed: bool
    reversed: bool
    counts: str
    proposed: bool = False


def edge_style(link: dict) -> tuple[str, bool]:
    """(colour, dashed) for one link's evidence. The whole colour rule is here.

    Contradiction wins over support: a link with both a supporting and an
    opposing finding is red, because the disagreement is the thing a reader
    most needs to see. Its dash still reports the source breadth.
    """
    against = list(link.get("no") or []) + list(link.get("no_effect") or [])
    for_it = list(link.get("yes") or [])
    if against:
        colour = CONTRADICTED
    elif for_it:
        colour = SUPPORTED
    else:
        colour = UNEVIDENCED
    dashed = link.get("state") == "single_source"
    return colour, dashed


def _counts(link: dict) -> str:
    yes = len(link.get("yes") or [])
    no = len(link.get("no") or [])
    none = len(link.get("no_effect") or [])
    parts = []
    if yes:
        parts.append(f"{yes}✓")
    if no:
        parts.append(f"{no}✗")
    if none:
        parts.append(f"{none}∅")
    return " ".join(parts)


def _collect(record: Bundle) -> tuple[dict[str, _Node], list[_Edge]]:
    nodes: dict[str, _Node] = {}
    edges: dict[str, _Edge] = {}

    for hypothesis in record.hypotheses:
        links = hypothesis.evidence.get("links") or {}
        nodes.setdefault(
            hypothesis.subject,
            _Node(id=hypothesis.subject, name=hypothesis.subject_name),
        )
        depth = 0
        for step in hypothesis.path:
            # An `analogical_transfer` path is the *donor's* edge, not the
            # subject's, so its `from` is a node no trace ever arrives at. Left
            # uncollected, the edge would be dropped for want of an endpoint and
            # the picture would show a floating node with nothing entering it.
            # Collect both ends of every step and the drawing stays honest.
            nodes.setdefault(
                step["from"], _Node(id=step["from"], name=step["from_name"])
            )
            target = step["to"]
            node = nodes.setdefault(target, _Node(id=target, name=step["to_name"]))
            depth += 1
            # Longest path wins, so an edge always points forward across
            # columns when the traces agree about ordering.
            node.col = max(node.col, depth)
            link_id = step["link"]
            if link_id not in edges:
                colour, dashed = edge_style(links.get(link_id) or {})
                edges[link_id] = _Edge(
                    src=step["from"],
                    dst=target,
                    link=link_id,
                    how=step["how"],
                    colour=colour,
                    dashed=dashed,
                    reversed=bool(step["reversed"]),
                    counts=_counts(links.get(link_id) or {}),
                )

        # The claim itself: subject → object is the edge the graph does *not*
        # have, which is what makes this a hypothesis rather than a lookup.
        # Drawing it distinctly is what stops the analogical case from reading
        # as "somebody showed nintedanib does this, so the picture is about
        # nintedanib" -- and it is the only edge here backed by no evidence at
        # all, so it is grey by the same rule as any unevidenced link.
        claim_key = f"claim:{hypothesis.id}"
        # If a real link already joins the two ends, the claim edge would draw
        # on top of it and assert "unevidenced" over an evidenced line. (That
        # combination means the relation is already stated, which validation
        # rejects upstream -- but a picture must not depend on that holding.)
        already_drawn = any(
            e.src == hypothesis.subject and e.dst == hypothesis.object
            for e in edges.values()
        )
        if (
            claim_key not in edges
            and not already_drawn
            and hypothesis.object != hypothesis.subject
        ):
            nodes.setdefault(
                hypothesis.object,
                _Node(id=hypothesis.object, name=hypothesis.object_name),
            )
            edges[claim_key] = _Edge(
                src=hypothesis.subject,
                dst=hypothesis.object,
                link=hypothesis.id,
                how=hypothesis.motif.replace("_", " "),
                colour=UNEVIDENCED,
                dashed=True,
                reversed=False,
                counts="",
                proposed=True,
            )

    _order_rows(nodes, list(edges.values()))
    for node in nodes.values():
        node.lines = _wrap(node.name)
    return nodes, list(edges.values())


def _order_rows(nodes: dict[str, _Node], edges: list[_Edge], sweeps: int = 6) -> None:
    """Order each column so edges cross as little as possible.

    First-seen order is what a reader gets otherwise, and on the demo record it
    put every trace's own nodes on different rows: three traces, and every
    single edge swept diagonally across all of them. This is the barycentre
    heuristic -- repeatedly move each node to the average row of its
    neighbours in the previous column, alternating direction. It is not
    optimal (minimising crossings is NP-hard) and does not need to be; it is
    deterministic, ties break on node id, and it turns a knot into a picture.
    """
    by_col: dict[int, list[_Node]] = {}
    for node in nodes.values():
        by_col.setdefault(node.col, []).append(node)
    for column in by_col.values():
        for row, node in enumerate(column):
            node.row = row

    for sweep in range(sweeps):
        forward = sweep % 2 == 0
        for col in sorted(by_col, reverse=not forward):
            for node in by_col[col]:
                neighbours = [
                    nodes[other].row
                    for edge in edges
                    for other, mine in ((edge.src, edge.dst), (edge.dst, edge.src))
                    if mine == node.id
                    and other in nodes
                    and (
                        nodes[other].col < col if forward else nodes[other].col > col
                    )
                ]
                node.bary = (
                    sum(neighbours) / len(neighbours) if neighbours else float(node.row)
                )
            by_col[col].sort(key=lambda n: (n.bary, n.id))
            for row, node in enumerate(by_col[col]):
                node.row = row


def _edge_path(a: _Node, b: _Node) -> tuple[str, float, float]:
    """A gentle cubic from a's right edge to b's left edge, plus its midpoint.

    Curved rather than straight so that two edges between the same columns do
    not overlay each other, and so a same-column edge still has somewhere to go.
    """
    x1, y1 = a.x + _BOX_W, a.cy
    x2, y2 = b.x, b.cy
    if b.col <= a.col:  # backwards or sideways: leave and enter on the same side
        x2 = b.x + _BOX_W
        bend = 60.0
        return f"M{x1},{y1} C{x1 + bend},{y1} {x2 + bend},{y2} {x2},{y2}", (
            max(x1, x2) + bend * 0.7
        ), (y1 + y2) / 2
    bend = max(28.0, (x2 - x1) / 2)
    return (
        f"M{x1},{y1} C{x1 + bend},{y1} {x2 - bend},{y2} {x2},{y2}",
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def _free_label_y(
    placed: list[tuple[float, float, float]], x: float, y: float, chars: int
) -> float:
    """Push a label down until it clears the ones already written.

    Two edges that cross have the same midpoint, so their labels land on the
    same pixel and overprint into gibberish -- on the demo record ``L1 inhibits``
    and ``L5 activates`` came out as one unreadable smear. Alternating an
    offset is not enough (the colliding pair can share a parity); this checks
    what has actually been placed.
    """
    half = chars * 2.9 + 6
    step = 14.0
    for _ in range(8):
        clash = any(
            abs(y - py) < 12 and (x - half) < (px + phalf) and (px - phalf) < (x + half)
            for px, py, phalf in placed
        )
        if not clash:
            break
        y += step
    placed.append((x, y, half))
    return y


def _marker(colour: str, name: str) -> str:
    return (
        f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{colour}"/></marker>'
    )


_MARKER_IDS = {SUPPORTED: "arrow-blue", CONTRADICTED: "arrow-red",
               UNEVIDENCED: "arrow-grey"}


def to_svg(record: Bundle) -> str:
    """One SVG for the whole record. Deterministic: same record, same bytes."""
    nodes, edges = _collect(record)
    if not nodes:
        nodes = {}
    cols = max((n.col for n in nodes.values()), default=0) + 1
    rows = max((n.row for n in nodes.values()), default=0) + 1
    row_h = _MAX_LINES * _LINE_H + 16 + _ROW_GAP
    width = _PAD * 2 + (cols - 1) * _COL_W + _BOX_W
    height = _PAD * 2 + _TITLE_H + rows * row_h + _LEGEND_H

    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, '
        f'Helvetica, Arial, sans-serif">',
        "<defs>"
        + "".join(_marker(c, n) for c, n in sorted(_MARKER_IDS.items()))
        + "</defs>",
        f'<rect width="100%" height="100%" fill="{_PAPER}"/>',
    ]

    # Title: which graph, which round, and the question it was asked.
    out.append(
        f'<text x="{_PAD}" y="{_PAD + 6}" font-size="15" font-weight="600" '
        f'fill="{_INK}">{escape(record.graph_id)} · round {record.round} · '
        f"{len(record.hypotheses)} trace"
        f'{"" if len(record.hypotheses) == 1 else "s"}</text>'
    )
    if record.question:
        out.append(
            f'<text x="{_PAD}" y="{_PAD + 27}" font-size="12" fill="{_MUTED}">'
            f"{escape(_wrap(record.question, width - _PAD * 2)[0])}</text>"
        )

    # Edges first so the node boxes sit on top of the curve ends. Labels ride
    # the curve via textPath rather than floating at a computed midpoint: a
    # long sweep between distant rows passes nowhere near the straight-line
    # midpoint, and a label parked there lands on somebody else's edge.
    placed: list[tuple[float, float, float]] = []
    for edge in edges:
        a, b = nodes.get(edge.src), nodes.get(edge.dst)
        if not (a and b):
            # Unreachable now that both endpoints of every step are collected,
            # and it stays a hard failure rather than a silently missing edge.
            raise ValueError(f"edge {edge.link} has an endpoint not in the record")
        path, mx, my = _edge_path(a, b)
        dash = ' stroke-dasharray="6 4"' if edge.dashed else ""
        if edge.proposed:
            dash = ' stroke-dasharray="2 5"'
        width = 1.5 if edge.proposed else 2
        out.append(
            f'<path d="{path}" fill="none" '
            f'stroke="{edge.colour}" stroke-width="{width}"{dash} '
            f'marker-end="url(#{_MARKER_IDS[edge.colour]})"/>'
        )
        if edge.proposed:
            label = f"proposed: {edge.how}"
        else:
            label = f"{edge.link} {edge.how}"
            if edge.reversed:
                label += " (reversed)"
            if edge.counts:
                label += f"  {edge.counts}"
        # Horizontal, not riding the curve: a textPath label on a steep sweep
        # comes out rotated to near-vertical and stops being readable, which is
        # a strange price to pay for sitting exactly on its line. Halo via
        # paint-order rather than a background rectangle, which would blank out
        # whatever passes underneath.
        label_y = _free_label_y(placed, mx, my - 10, len(label))
        out.append(
            f'<text x="{mx:.1f}" y="{label_y:.1f}" font-size="10.5" '
            f'text-anchor="middle" fill="{edge.colour}" stroke="{_PAPER}" '
            f'stroke-width="3.5" paint-order="stroke" stroke-linejoin="round">'
            f"{escape(label)}</text>"
        )

    for node in sorted(nodes.values(), key=lambda n: (n.col, n.row)):
        out.append(
            f'<rect x="{node.x:.0f}" y="{node.y:.0f}" width="{_BOX_W}" '
            f'height="{node.height:.0f}" rx="7" fill="{_NODE_FILL}" '
            f'stroke="{_NODE_STROKE}"/>'
        )
        for i, line in enumerate(node.lines):
            out.append(
                f'<text x="{node.cx:.0f}" y="{node.y + 20 + i * _LINE_H:.0f}" '
                f'font-size="12.5" text-anchor="middle" fill="{_INK}">'
                f"{escape(line)}</text>"
            )
        out.append(
            f'<text x="{node.cx:.0f}" y="{node.y + node.height + 12:.0f}" '
            f'font-size="10" text-anchor="middle" fill="{_MUTED}">'
            f"{escape(node.id)}</text>"
        )

    # The legend is not decoration: the colours are a claim about evidence and
    # an unlabelled red line is an unsourced assertion.
    ly = _PAD + _TITLE_H + rows * row_h + 12
    entries = [
        (SUPPORTED, "every finding supports it"),
        (CONTRADICTED, "a finding argues against it (no / no_effect)"),
        (UNEVIDENCED, "no verbatim finding"),
    ]
    x = _PAD
    for colour, text in entries:
        out.append(
            f'<line x1="{x}" y1="{ly}" x2="{x + 26}" y2="{ly}" '
            f'stroke="{colour}" stroke-width="2"/>'
            f'<text x="{x + 32}" y="{ly + 4}" font-size="11" fill="{_INK}">'
            f"{escape(text)}</text>"
        )
        x += 42 + len(text) * 5.9
    out.append(
        f'<line x1="{_PAD}" y1="{ly + 22}" x2="{_PAD + 26}" y2="{ly + 22}" '
        f'stroke="{_MUTED}" stroke-width="2" stroke-dasharray="6 4"/>'
        f'<text x="{_PAD + 32}" y="{ly + 26}" font-size="11" fill="{_INK}">'
        "dashed: single source · solid: more than one · ✓ supporting / "
        "✗ opposing / ∅ no-effect findings</text>"
    )
    out.append(
        f'<line x1="{_PAD}" y1="{ly + 44}" x2="{_PAD + 26}" y2="{ly + 44}" '
        f'stroke="{UNEVIDENCED}" stroke-width="1.5" stroke-dasharray="2 5"/>'
        f'<text x="{_PAD + 32}" y="{ly + 48}" font-size="11" fill="{_INK}">'
        "finely dotted: the hypothesis itself — the edge the graph does not "
        "have. On an analogical transfer the solid edge belongs to the "
        "analogue, not to the subject.</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"
