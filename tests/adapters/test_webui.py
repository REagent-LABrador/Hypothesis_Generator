"""The web UI adapter: trace strings, metrics, and id-referenced one-liners.

The property under test throughout: nothing on a card states anything the
record does not, and everything a card states can be walked back to an id.
"""

from __future__ import annotations

import json
from pathlib import Path

from adapters.webui import payload as webui
from hyp_gen.cli import main
from hyp_gen.graph import KnowledgeGraph
from conftest import bundle
from hyp_gen.pipeline import Generator
from hyp_gen.hypothesis import Hypothesis, ValidationIssue

GRAPH = Path(__file__).resolve().parents[2] / "examples" / "knowledge-graph.json"


def _record(graph: KnowledgeGraph, params):
    return bundle(Generator(graph=graph, params=params).run())


# -- the trace string ------------------------------------------------------


def test_trace_renders_the_walk_as_one_string() -> None:
    hypothesis = Hypothesis(
        id="H-x", motif="transitive_chain", subject="t1", object="t5",
        subject_name="pirfenidone", object_name="idiopathic pulmonary fibrosis",
        hops=2,
        path=[
            {"link": "L14", "from": "t1", "from_name": "pirfenidone",
             "how": "inhibits", "to": "t4",
             "to_name": "myofibroblast differentiation",
             "reversed": False, "state": "single_source", "support": 0.5},
            {"link": "L4", "from": "t4",
             "from_name": "myofibroblast differentiation",
             "how": "contributes_to", "to": "t5",
             "to_name": "idiopathic pulmonary fibrosis",
             "reversed": False, "state": "single_source", "support": 0.6},
        ],
    )
    assert webui.trace(hypothesis) == (
        "pirfenidone --inhibits--> myofibroblast differentiation "
        "--contributes_to--> idiopathic pulmonary fibrosis"
    )


def test_trace_keeps_a_reversed_hop_pointing_the_right_way() -> None:
    """Flattening a reversed edge to --> would present a walk against the
    graph's stated arrow as a walk along it."""
    hypothesis = Hypothesis(
        id="H-x", motif="transitive_chain", subject="a", object="c",
        subject_name="A", object_name="C", hops=1,
        path=[{"link": "L1", "from": "a", "from_name": "A", "how": "drives",
               "to": "c", "to_name": "C", "reversed": True,
               "state": "single_source", "support": 0.4}],
    )
    assert webui.trace(hypothesis) == "A <--drives-- C"


# -- metrics and payload shape ---------------------------------------------


def test_cards_carry_the_three_named_metrics_and_rank(
    graph: KnowledgeGraph, params
) -> None:
    payload = webui.emit(_record(graph, params))
    assert payload.hypotheses
    for card in payload.hypotheses:
        assert card.metrics.support is not None
        assert card.metrics.novelty is not None
        assert card.metrics.testability is not None
        assert card.metrics.rank is not None
        assert card.trace  # never an empty walk on a pathed hypothesis


def test_payload_round_trips_through_json(graph: KnowledgeGraph, params) -> None:
    """The payload is for a UI on the other side of a serialisation boundary."""
    payload = webui.emit(_record(graph, params))
    again = webui.WebPayload.model_validate(json.loads(payload.model_dump_json()))
    assert again == payload


def test_emit_is_pure(graph: KnowledgeGraph, params) -> None:
    record = _record(graph, params)
    assert webui.emit(record) == webui.emit(record)


# -- the one-liners --------------------------------------------------------


def test_every_ref_resolves_to_something_in_the_slate(
    graph: KnowledgeGraph, params
) -> None:
    """A one-liner's refs are its citations. A ref that resolves to nothing is
    a claim with a fake footnote, which is worse than no footnote."""
    record = _record(graph, params)
    payload = webui.emit(record)
    by_id = {h.id: h for h in record.hypotheses}
    for card in payload.hypotheses:
        hypothesis = by_id[card.id]
        known = set(hypothesis.evidence.get("links") or {})
        known |= set(hypothesis.evidence.get("findings") or {})
        known |= {step["link"] for step in hypothesis.path}
        gap = hypothesis.evidence.get("gap") or {}
        if gap.get("id"):
            known.add(gap["id"])
        for highlight in card.highlights:
            for ref in highlight.refs:
                assert ref in known, (card.id, highlight.text, ref)


def test_support_lines_name_single_source_as_such(
    graph: KnowledgeGraph, params
) -> None:
    """"Backed by f16" reads very differently once you know f16 is the only
    source there is, so the state is part of the sentence."""
    record = _record(graph, params)
    payload = webui.emit(record)
    for card, hypothesis in zip(payload.hypotheses, record.hypotheses):
        single = [s for s in hypothesis.path if s["state"] == "single_source"]
        if not single:
            continue
        support_text = " ".join(
            h.text for h in card.highlights if h.kind == "support"
        )
        assert "One source only" in support_text, card.id


def test_a_disagreed_link_produces_a_contradiction_line_with_the_quote() -> None:
    hypothesis = Hypothesis(
        id="H-x", motif="transitive_chain", subject="a", object="c",
        subject_name="metformin", object_name="inflammation", hops=1,
        path=[{"link": "L6", "from": "a", "from_name": "metformin",
               "how": "suppresses", "to": "c", "to_name": "inflammation",
               "reversed": False, "state": "disagreed", "support": 0.3}],
        evidence={
            "links": {"L6": {"yes": ["f5"], "no": ["f6"], "no_effect": [],
                             "stated_confidence": 0.5,
                             "recomputed_support": 0.3}},
            "findings": {
                "f5": {"quote": "metformin reduced markers", "hedged": False},
                "f6": {"quote": "no change in inflammatory markers was seen",
                       "hedged": False},
            },
        },
    )
    card = webui._card(hypothesis, frozenset())
    contradictions = [h for h in card.highlights if h.kind == "contradiction"]
    assert len(contradictions) == 1
    assert "f6" in contradictions[0].refs and "L6" in contradictions[0].refs
    assert "no change in inflammatory markers" in contradictions[0].text
    # The recomputation drifted 0.2 below the stated confidence: said out loud.
    cautions = " ".join(h.text for h in card.highlights if h.kind == "caution")
    assert "0.30" in cautions and "0.50" in cautions


def test_novelty_prefers_the_recorded_gap_note(
    graph: KnowledgeGraph, params
) -> None:
    """When the graph builder wrote down why the pair is unstated, that sentence is the
    novelty line — recorded beats templated."""
    record = _record(graph, params)
    payload = webui.emit(record)
    for card, hypothesis in zip(payload.hypotheses, record.hypotheses):
        novelty = [h for h in card.highlights if h.kind == "novelty"]
        assert len(novelty) == 1, card.id
        gap = hypothesis.evidence.get("gap") or {}
        if gap.get("id") and gap.get("note"):
            assert gap["id"] in novelty[0].refs
            assert gap["note"][:40] in novelty[0].text


def test_highlights_lead_with_the_weakest_kind(
    graph: KnowledgeGraph, params
) -> None:
    """Support last: a card that leads with its support reads as advocacy."""
    payload = webui.emit(_record(graph, params))
    for card in payload.hypotheses:
        order = [webui._KIND_ORDER[h.kind] for h in card.highlights]
        assert order == sorted(order), card.id


# -- the safety contract ---------------------------------------------------


def test_a_rejected_hypothesis_is_flagged_on_its_card(
    graph: KnowledgeGraph, params
) -> None:
    """Same contract as the report modes: a mode changes the form, never the
    safety. The payload must name a rejection in both flags and highlights."""
    record = _record(graph, params)
    record.hypotheses[0].issues.append(
        ValidationIssue(
            code="illegal_citation",
            detail="cites L99, which was not in its evidence pack",
            severity="error",
        )
    )
    card = webui.emit(record).hypotheses[0]
    assert any("CITATION REJECTED" in f for f in card.status.flags)
    failures = [h for h in card.highlights if h.kind == "failure"]
    assert any("CITATION REJECTED" in h.text for h in failures)
    assert any("illegal_citation" in h.text for h in failures)
    # And weakest-first ordering puts them at the top of the card.
    assert card.highlights[0].kind == "failure"


def test_truncated_coverage_puts_the_absence_warning_on_the_payload(
    graph: KnowledgeGraph, params
) -> None:
    """Novelty one-liners are only readable next to this; it is payload-level
    so no card view can drop it."""
    payload = webui.emit(_record(graph, params))
    assert any("not evidence of absence" in w for w in payload.warnings)


def test_shared_caveats_move_to_payload_warnings_not_every_card(
    graph: KnowledgeGraph, params
) -> None:
    record = _record(graph, params)
    if len(record.hypotheses) < 2:
        return
    shared = frozenset.intersection(*(frozenset(h.caveats) for h in record.hypotheses))
    assert shared, "fixture must have a caveat common to every hypothesis"
    payload = webui.emit(record)
    for caveat in shared:
        opening = caveat.split(":")[0][:40]
        assert any(opening in w for w in payload.warnings), caveat
        for card in payload.hypotheses:
            assert not any(opening in h.text for h in card.highlights), card.id


# -- the CLI ---------------------------------------------------------------

