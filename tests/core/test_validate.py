"""Validation: the graph checks the structure, and the pack checks the model."""

from __future__ import annotations

from hyp_gen.generate.candidates import Candidate, enumerate_candidates
from hyp_gen.generate.evidence import build_pack
from hyp_gen.graph import Edge, GraphIndex
from hyp_gen.params import Params
from hyp_gen.hypothesis import Articulation, Claim, Critique, CritiqueFinding
from hyp_gen.generate.scoring import score_candidate
from hyp_gen.checks.validate import check_citations, check_structure


def _pack_for(index: GraphIndex, params: Params, motif: str):
    candidate = next(
        c for c in enumerate_candidates(index, params) if c.motif == motif
    )
    scores = score_candidate(index, candidate, params)
    return candidate, build_pack(index, candidate, scores)


def test_clean_candidates_pass(index: GraphIndex, params: Params) -> None:
    candidate, pack = _pack_for(index, params, "transitive_chain")
    assert [i for i in check_structure(index, candidate, pack) if i.severity == "error"] == []


def test_restating_a_stated_link_is_blocked(index: GraphIndex, params: Params) -> None:
    """A hypothesis that restates a finding is a summary, not a hypothesis."""
    edge = next(e for e in index.neighbors("t1") if e.link_id == "L1")
    candidate = Candidate(
        id="H-bad", motif="transitive_chain", subject="t1", object="t3", path=(edge,)
    )
    pack = build_pack(index, candidate, score_candidate(index, candidate, params))
    codes = {i.code for i in check_structure(index, candidate, pack) if i.severity == "error"}
    assert "already_stated" in codes


def test_broken_paths_are_errors(index: GraphIndex, params: Params) -> None:
    """A chain that does not connect means the enumerator and the graph
    disagree, which is a bug worth failing loudly on."""
    disconnected = Edge(link_id="L8", src="t2", dst="t11", how="slows", forward=True)
    candidate = Candidate(
        id="H-broken",
        motif="transitive_chain",
        subject="t8",
        object="t11",
        path=(disconnected,),
    )
    pack = build_pack(index, candidate, score_candidate(index, candidate, params))
    codes = {i.code for i in check_structure(index, candidate, pack)}
    assert "broken_path" in codes


def test_unknown_ids_are_errors(index: GraphIndex, params: Params) -> None:
    candidate = Candidate(id="H-ghost", motif="gap_closure", subject="t404", object="t5")
    pack = build_pack(index, candidate, score_candidate(index, candidate, params))
    codes = {i.code for i in check_structure(index, candidate, pack) if i.severity == "error"}
    assert "unknown_thing" in codes


def test_a_gap_with_no_path_is_flagged_but_not_blocked(index: GraphIndex, params: Params) -> None:
    candidate = Candidate(id="H-lonely", motif="gap_closure", subject="t1", object="t9")
    pack = build_pack(index, candidate, score_candidate(index, candidate, params))
    issues = check_structure(index, candidate, pack)
    assert "unconnected_gap" in {i.code for i in issues}
    assert all(i.severity != "error" for i in issues if i.code == "unconnected_gap")


def test_citations_outside_the_pack_are_rejected(index: GraphIndex, params: Params) -> None:
    """The whole point of the pack: a model that cites what it was never shown
    has stopped reporting and started remembering."""
    _, pack = _pack_for(index, params, "transitive_chain")
    articulation = Articulation(
        statement="s",
        mechanism="m",
        claims=[Claim(text="c", cites=["L999"])],
        novel_because="n",
        falsifier="f",
        decisive_experiment="d",
    )
    issues = check_citations(pack, articulation, None)
    assert [i.code for i in issues] == ["illegal_citation"]
    assert issues[0].severity == "error"


def test_legal_citations_pass(index: GraphIndex, params: Params) -> None:
    _, pack = _pack_for(index, params, "transitive_chain")
    legal = sorted(pack.legal_ids())[0]
    articulation = Articulation(
        statement="s",
        mechanism="m",
        claims=[Claim(text="c", cites=[legal])],
        novel_because="n",
        falsifier="f",
        decisive_experiment="d",
    )
    assert check_citations(pack, articulation, None) == []


def test_uncited_claims_must_be_marked_inferred(index: GraphIndex, params: Params) -> None:
    _, pack = _pack_for(index, params, "transitive_chain")
    sloppy = Articulation(
        statement="s",
        mechanism="m",
        claims=[Claim(text="unsourced assertion", cites=[])],
        novel_because="n",
        falsifier="f",
        decisive_experiment="d",
    )
    honest = sloppy.model_copy(
        update={"claims": [Claim(text="a reasoning step", cites=[], inferred=True)]}
    )
    assert {i.code for i in check_citations(pack, sloppy, None)} == {"uncited_claim"}
    assert check_citations(pack, honest, None) == []


def test_critique_citations_are_audited_too(index: GraphIndex, params: Params) -> None:
    _, pack = _pack_for(index, params, "transitive_chain")
    critique = Critique(
        verdict="unsupported",
        strongest_objection="o",
        per_claim=[CritiqueFinding(claim_index=0, verdict="unsupported", reason="r", cites=["p999"])],
    )
    assert {i.code for i in check_citations(pack, None, critique)} == {"illegal_citation"}
