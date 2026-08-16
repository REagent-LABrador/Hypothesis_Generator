"""The trace diagram: what the colours claim, and that the SVG is well formed.

The colour rule is the whole point of this view, so most of these tests are
about `edge_style` rather than about pixels.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from adapters.webui import diagram
from hyp_gen.cli import main
from hyp_gen.graph import KnowledgeGraph
from hyp_gen.params import Params
from adapters.common import Bundle
from conftest import bundle
from hyp_gen.pipeline import Generator
from hyp_gen.hypothesis import Provenance, Hypothesis

GRAPH = Path(__file__).resolve().parents[2] / "examples" / "knowledge-graph.json"


# -- the colour rule -------------------------------------------------------


def test_a_link_with_only_supporting_findings_is_blue() -> None:
    colour, dashed = diagram.edge_style(
        {"yes": ["f1", "f13"], "no": [], "no_effect": [], "state": "agreed"}
    )
    assert colour == diagram.SUPPORTED
    assert not dashed  # more than one source


def test_a_link_with_an_opposing_finding_is_red() -> None:
    colour, _ = diagram.edge_style(
        {"yes": ["f6"], "no": ["f7"], "no_effect": [], "state": "disagreed"}
    )
    assert colour == diagram.CONTRADICTED


def test_contradiction_wins_over_support() -> None:
    """A link that is mostly supported but has one dissenting finding is red.

    The disagreement is the thing a reader most needs to see; averaging it away
    into blue would be the picture telling a nicer story than the graph does.
    """
    colour, _ = diagram.edge_style(
        {"yes": ["f1", "f2", "f3"], "no": ["f4"], "no_effect": [], "state": "disagreed"}
    )
    assert colour == diagram.CONTRADICTED


def test_a_no_effect_finding_counts_as_contradiction() -> None:
    """"We looked and saw nothing" argues against a claimed effect."""
    colour, _ = diagram.edge_style(
        {"yes": [], "no": [], "no_effect": ["f9"], "state": "no_effect"}
    )
    assert colour == diagram.CONTRADICTED


def test_a_link_with_no_findings_is_grey_not_blue() -> None:
    """Unevidenced is not the same as supported, and must not look like it."""
    colour, _ = diagram.edge_style({"yes": [], "no": [], "no_effect": []})
    assert colour == diagram.UNEVIDENCED


def test_single_source_is_dashed_independently_of_colour() -> None:
    """Colour reports direction of evidence, dash reports breadth of it."""
    for evidence in ({"yes": ["f1"]}, {"no": ["f2"]}, {}):
        _, dashed = diagram.edge_style({**evidence, "state": "single_source"})
        assert dashed
    _, dashed = diagram.edge_style({"yes": ["f1"], "state": "agreed"})
    assert not dashed


# -- the rendered document -------------------------------------------------


def _record(graph: KnowledgeGraph, params) -> RunResult:
    return bundle(Generator(graph=graph, params=params).run())


def test_svg_is_well_formed_and_self_contained(
    graph: KnowledgeGraph, params
) -> None:
    svg = diagram.to_svg(_record(graph, params))
    root = ElementTree.fromstring(svg)  # raises if malformed
    assert root.tag.endswith("svg")
    # A static image that fetches from the network is not a static image.
    assert "http://www.w3.org/2000/svg" in svg
    assert "<image" not in svg and "xlink:href" not in svg


def test_every_node_and_link_in_every_trace_is_drawn(
    graph: KnowledgeGraph, params
) -> None:
    record = _record(graph, params)
    svg = diagram.to_svg(record)
    for hypothesis in record.hypotheses:
        assert hypothesis.subject_name.split()[0] in svg
        for step in hypothesis.path:
            assert step["link"] in svg


def test_no_edge_is_silently_dropped_for_a_missing_endpoint(
    graph: KnowledgeGraph,
) -> None:
    """An `analogical_transfer` path is the *donor's* edge, so its source is a
    node no trace ever arrives at. The first version of this renderer dropped
    exactly those edges and drew the target floating with nothing entering it."""
    # `repurposing` is the profile that reaches for a cross-molecule analogy.
    record = _record(graph, Params.profile("repurposing"))
    analogical = [h for h in record.hypotheses if h.motif == "analogical_transfer"]
    assert analogical, "fixture must produce an analogical transfer"

    nodes, edges = diagram._collect(record)
    for edge in edges:
        assert edge.src in nodes, edge.link
        assert edge.dst in nodes, edge.link

    # Every node has something touching it: no orphans in the picture.
    touched = {e.src for e in edges} | {e.dst for e in edges}
    assert set(nodes) <= touched


def test_the_hypothesis_itself_is_drawn_as_a_proposed_edge(
    graph: KnowledgeGraph,
) -> None:
    """On an analogical transfer the solid edge belongs to the analogue. Without
    the proposed edge drawn separately, the picture reads as a claim about the
    analogue rather than about the subject."""
    record = _record(graph, Params.profile("repurposing"))
    hypothesis = next(h for h in record.hypotheses if h.motif == "analogical_transfer")
    _, edges = diagram._collect(record)

    # Match on the hypothesis id: two hypotheses here share the subject
    # pirfenidone, so filtering by source alone finds both their claim edges.
    claim = [e for e in edges if e.proposed and e.link == hypothesis.id]
    assert len(claim) == 1
    assert claim[0].src == hypothesis.subject
    assert claim[0].dst == hypothesis.object
    # Unevidenced by construction: no finding backs an edge the graph lacks.
    assert claim[0].colour == diagram.UNEVIDENCED

    svg = diagram.to_svg(record)
    assert "proposed: analogical transfer" in svg
    assert "the edge the graph does not have" in svg


def test_no_proposed_edge_where_a_real_link_already_joins_the_ends() -> None:
    """It would draw an unevidenced line on top of an evidenced one."""
    hypothesis = Hypothesis(
        id="H-x", motif="transitive_chain", subject="a", object="b",
        subject_name="A", object_name="B", hops=1,
        path=[{"link": "L1", "from": "a", "from_name": "A", "how": "inhibits",
               "to": "b", "to_name": "B", "reversed": False,
               "state": "single_source", "support": 0.5}],
        evidence={"links": {"L1": {"yes": ["f1"], "no": [], "no_effect": [],
                                   "state": "single_source"}}},
    )
    record = Bundle(
        provenance=Provenance(graph_id="g", round=1, question="q"), hypotheses=[hypothesis])
    _, edges = diagram._collect(record)
    assert [e.link for e in edges] == ["L1"]


def test_nodes_are_deduplicated_across_traces(
    graph: KnowledgeGraph, params
) -> None:
    """Two traces crossing one node should converge on it, not draw it twice --
    that convergence is a fact about the graph worth seeing."""
    record = _record(graph, params)
    shared = None
    for i, first in enumerate(record.hypotheses):
        for second in record.hypotheses[i + 1 :]:
            crossing = {s["to"] for s in first.path} & {s["to"] for s in second.path}
            if crossing:
                shared = sorted(crossing)[0]
                break
    assert shared, "fixture must have two traces crossing a node"

    nodes, _ = diagram._collect(record)
    assert len([n for n in nodes.values() if n.id == shared]) == 1
    # ...and the id caption for it appears exactly once in the document.
    svg = diagram.to_svg(record)
    assert svg.count(f">{shared}</text>") == 1


def test_the_legend_states_what_each_colour_claims(
    graph: KnowledgeGraph, params
) -> None:
    """An unlabelled red line is an unsourced assertion."""
    svg = diagram.to_svg(_record(graph, params))
    assert diagram.SUPPORTED in svg and diagram.CONTRADICTED in svg
    assert "every finding supports it" in svg
    assert "argues against it" in svg
    assert "single source" in svg


def test_a_contradicted_edge_renders_red_end_to_end() -> None:
    """The demo graph's L6 is the one disagreed link; a trace through it must
    come out red."""
    hypothesis = Hypothesis(
        id="H-x", motif="transitive_chain", subject="t9", object="t7",
        subject_name="AMPK", object_name="inflammation", hops=1,
        path=[{"link": "L6", "from": "t9", "from_name": "AMPK",
               "how": "suppresses", "to": "t7", "to_name": "inflammation",
               "reversed": False, "state": "disagreed", "support": 0.3}],
        evidence={"links": {"L6": {"yes": ["f6"], "no": ["f7"],
                                   "no_effect": [], "state": "disagreed"}}},
    )
    record = Bundle(
        provenance=Provenance(graph_id="g", round=1, question="q"), hypotheses=[hypothesis])
    svg = diagram.to_svg(record)
    # Check the drawn edges, not the whole document: the legend necessarily
    # contains every colour, so a document-wide search proves nothing. The one
    # graph edge here is red; the subject→object claim edge rides alongside it
    # in unevidenced grey.
    root = ElementTree.fromstring(svg)
    strokes = [
        el.get("stroke")
        for el in root.iter("{http://www.w3.org/2000/svg}path")
        if el.get("stroke")
    ]
    assert diagram.CONTRADICTED in strokes
    assert diagram.SUPPORTED not in strokes
    # The counts are on the label so a reader can see it is 1 for, 1 against.
    assert "1✓ 1✗" in svg


def test_text_is_escaped() -> None:
    """A node name with an ampersand must not produce invalid XML."""
    hypothesis = Hypothesis(
        id="H-x", motif="transitive_chain", subject="a", object="b",
        subject_name="tumour necrosis factor & friends <alpha>",
        object_name="B", hops=1,
        path=[{"link": "L1", "from": "a",
               "from_name": "tumour necrosis factor & friends <alpha>",
               "how": "inhibits", "to": "b", "to_name": "B",
               "reversed": False, "state": "single_source", "support": 0.5}],
    )
    record = Bundle(
        provenance=Provenance(graph_id="g&1", round=1, question="a < b?"),
                  hypotheses=[hypothesis])
    ElementTree.fromstring(diagram.to_svg(record))  # raises if unescaped


def test_render_is_deterministic(graph: KnowledgeGraph, params) -> None:
    record = _record(graph, params)
    assert diagram.to_svg(record) == diagram.to_svg(record)


def test_an_empty_slate_still_renders(graph: KnowledgeGraph) -> None:
    """A record with nothing in it is a real outcome, not a crash."""
    record = Bundle(
        provenance=Provenance(graph_id="g", round=1, question="q"), hypotheses=[])
    ElementTree.fromstring(diagram.to_svg(record))


# -- the CLI ---------------------------------------------------------------

