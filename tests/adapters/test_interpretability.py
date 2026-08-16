"""The shared LABrador interpretability contract, as the webui payload carries it.

The property under test throughout: everything in ``interpretability`` is a
mapping of the document — never a recomputation that could drift, never a claim
the document does not carry — and the block is complete enough that a UI can
answer what was concluded, why, from which evidence, and what would change it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from adapters import interpretability as interp
from adapters.webui import payload as webui
from conftest import bundle
from hyp_gen.graph import KnowledgeGraph
from hyp_gen.pipeline import Generator

ROOT = Path(__file__).resolve().parents[2]
DEMO_CARDS = ROOT / "examples" / "cards.json"
CARDS_SCHEMA = ROOT / "schemas" / "cards.schema.json"
INTERP_SCHEMA = ROOT / "schemas" / "interpretability.schema.json"


@pytest.fixture
def record(graph: KnowledgeGraph, params):
    """A dry run: deterministic, no model calls — the floor every run meets."""
    return bundle(Generator(graph=graph, params=params).run())


@pytest.fixture
def block(record) -> interp.Interpretability:
    return webui.emit(record).interpretability


# -- presence and schema ----------------------------------------------------


def test_every_successful_payload_carries_interpretability(record) -> None:
    payload = webui.emit(record)
    assert payload.interpretability is not None
    assert payload.interpretability.schema_version == "1.0.0"


def test_removing_interpretability_fails_validation(record) -> None:
    """The field is required, not merely present: a payload without it must
    not validate against the payload contract."""
    dumped = json.loads(webui.emit(record).model_dump_json())
    del dumped["interpretability"]
    with pytest.raises(Exception):
        webui.WebPayload.model_validate(dumped)


def test_the_checked_in_example_carries_it_and_validates() -> None:
    payload = webui.WebPayload.model_validate(json.loads(DEMO_CARDS.read_text()))
    assert payload.interpretability.headline.title


def test_the_checked_in_example_is_not_stale() -> None:
    """The example is what emit() writes for examples/hypothesis.json, exactly —
    a drifted example documents a contract nobody honours."""
    from adapters.common import load

    rebuilt = webui.emit(load(ROOT / "examples" / "hypothesis.json"))
    assert rebuilt == webui.WebPayload.model_validate(json.loads(DEMO_CARDS.read_text())), (
        "run `hypwebui examples/hypothesis.json --cards examples/cards.json`"
    )


def test_published_schemas_require_interpretability() -> None:
    cards = json.loads(CARDS_SCHEMA.read_text())
    assert "interpretability" in cards["required"]
    standalone = json.loads(INTERP_SCHEMA.read_text())
    assert standalone["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "headline" in standalone["required"]


# -- reference integrity ----------------------------------------------------


def test_ids_are_unique_within_each_collection(block) -> None:
    for items in (block.metrics, block.steps, block.evidence, block.assumptions):
        ids = [item.id for item in items]
        assert len(ids) == len(set(ids)), ids


def test_every_reference_resolves(block) -> None:
    evidence_ids = {e.id for e in block.evidence}
    assumption_ids = {a.id for a in block.assumptions}
    metric_ids = {m.id for m in block.metrics}
    for owner in (*block.metrics, *block.steps):
        for ref in owner.evidence_ids:
            assert ref in evidence_ids, (owner.id, ref)
        for ref in owner.assumption_ids:
            assert ref in assumption_ids, (owner.id, ref)
    for interval in block.uncertainty.intervals:
        assert interval.metric_id in metric_ids


def test_ids_are_stable_not_positional(block) -> None:
    """An id derived from array position changes meaning on every reorder."""
    for metric in block.metrics:
        assert metric.id.startswith("metric."), metric.id
        assert not metric.id.split(".")[-1].isdigit(), metric.id
    for item in block.evidence:
        # Evidence ids carry the document's own finding ids.
        assert item.id.startswith("evidence.f"), item.id


# -- units, tagging, honesty ------------------------------------------------


def test_numeric_metrics_carry_units(block) -> None:
    for metric in block.metrics:
        if metric.value is not None:
            assert metric.unit, metric.id


def test_metrics_reference_evidence_or_assumptions_or_are_flagged(block) -> None:
    codes = {lim.code for lim in block.limitations}
    for metric in block.metrics:
        if not (metric.evidence_ids or metric.assumption_ids):
            assert "UNTAGGED_VALUE" in codes, metric.id


def test_heuristic_scores_are_labeled_and_never_called_probabilities(block) -> None:
    for metric in block.metrics:
        text = f"{metric.meaning} {metric.label}".lower()
        assert "probability" not in text.replace("not a probability", ""), metric.id
        if metric.unit == "score":
            assert "heuristic" in metric.meaning.lower(), metric.id


def test_output_is_strict_json_without_nan_or_infinity(record) -> None:
    dumped = webui.emit(record).model_dump(mode="json")
    text = json.dumps(dumped, allow_nan=False)  # raises on NaN/Infinity
    for metric in json.loads(text)["interpretability"]["metrics"]:
        if metric["value"] is not None:
            assert math.isfinite(metric["value"])


# -- reconstruction: no hidden scoring constant -----------------------------


def test_rank_score_reconstructs_exactly_from_the_emitted_step(record, block) -> None:
    """Every weight, prior and axis value the ranking used is in the step's
    inputs, so the displayed rank must fall out of them exactly — including
    the structure weight and the motif prior, which the params may not name."""
    step = next(s for s in block.steps if s.id == "step.rank_score")
    by_path = {i.path: i.value for i in step.inputs}
    winner = max(record.hypotheses, key=lambda h: h.rank_score)
    axes = ("support", "novelty", "testability", "contradiction_risk", "structure")
    base = sum(
        by_path[f"hypothesis.scores.{axis}"]
        * by_path[f"provenance.params.selection.rank_weights.{axis}"]
        for axis in axes
    )
    prior = by_path[f"provenance.params.motifs.weights.{winner.motif}"]
    assert round(base * prior, 4) == winner.rank_score == step.result.value


def test_per_link_support_reconstructs_from_emitted_weights(record, block) -> None:
    winner = max(record.hypotheses, key=lambda h: h.rank_score)
    per_link = winner.evidence.get("per_link_support") or []
    for entry in per_link:
        step = next(s for s in block.steps if s.id == f"step.link_support.{entry['link_id']}")
        by_path = {i.path: i.value for i in step.inputs}
        rebuilt = sum(
            by_path[f"provenance.params.evidence.support_weights.{part}"]
            * by_path[f"per_link_support.{entry['link_id']}.{part}"]
            for part in ("evidence_quality", "agreement", "independence")
        )
        if entry["capped"]:
            rebuilt = by_path["provenance.params.evidence.single_group_cap"]
        # Components are rounded to 3 decimals in the document; allow that.
        assert abs(rebuilt - entry["support"]) <= 0.002, entry["link_id"]
        assert step.result.value == entry["support"]


# -- unknowns stay null -----------------------------------------------------


def test_a_paper_without_a_doi_stays_null_and_is_flagged(record) -> None:
    winner = max(record.hypotheses, key=lambda h: h.rank_score)
    papers = winner.evidence.get("papers") or {}
    assert papers, "fixture must carry papers"
    for paper in papers.values():
        paper["doi"] = None
    block = webui.emit(record).interpretability
    for item in block.evidence:
        assert item.source_id is None
        assert item.source_url is None
    assert any(lim.code == "MISSING_SOURCE_IDENTIFIER" for lim in block.limitations)


def test_uncertainty_without_a_method_says_so(block) -> None:
    """No Monte Carlo ran, so there are no intervals to invent."""
    assert block.uncertainty.method == "none"
    assert block.uncertainty.intervals == []
    assert block.uncertainty.seed is None and block.uncertainty.draws is None
    assert block.uncertainty.limitations


# -- module-specific: hypothesis generator ----------------------------------


def test_dry_run_is_marked_and_qualified(record, block) -> None:
    assert block.extensions["run_mode"] == "DRY_RUN"
    assert block.headline.status in ("QUALIFIED", "INCONCLUSIVE")
    codes = {lim.code for lim in block.limitations}
    assert "STRUCTURAL_CANDIDATE_NOT_ARTICULATED" in codes


def test_the_graph_path_and_reversed_markers_survive(record, block) -> None:
    winner = max(record.hypotheses, key=lambda h: h.rank_score)
    path = block.extensions["graph_path"]
    assert [hop["link"] for hop in path] == [s["link"] for s in winner.path]
    assert [hop["reversed"] for hop in path] == [s["reversed"] for s in winner.path]


def test_verification_gates_survive_with_their_states(record, block) -> None:
    winner = max(record.hypotheses, key=lambda h: h.rank_score)
    gates = block.extensions["verification"]["gates"]
    assert [(g["name"], g["status"]) for g in gates] == [
        (g.name, g.status) for g in winner.verification.gates
    ]


def test_the_losing_candidates_are_ledgered_with_reasons(record, block) -> None:
    if len(record.hypotheses) < 2:
        pytest.skip("fixture produced a single hypothesis")
    winner = max(record.hypotheses, key=lambda h: h.rank_score)
    ledger = block.extensions["candidates"]
    assert {c["id"] for c in ledger} == {
        h.id for h in record.hypotheses if h.id != winner.id
    }
    for candidate in ledger:
        assert candidate["reason_not_selected"]
        assert candidate["rank_score"] is not None


def test_quotes_are_verbatim_from_the_document_never_invented(record, block) -> None:
    winner = max(record.hypotheses, key=lambda h: h.rank_score)
    findings = winner.evidence.get("findings") or {}
    for item in block.evidence:
        fid = item.id.removeprefix("evidence.")
        assert item.quote == (findings.get(fid) or {}).get("quote")
