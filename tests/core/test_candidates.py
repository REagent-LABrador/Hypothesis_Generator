"""Motif enumeration: the four structural reasons a hypothesis can exist."""

from __future__ import annotations

from hyp_gen.generate.candidates import enumerate_candidates
from hyp_gen.graph import GraphIndex
from hyp_gen.params import (
    FramingParams,
    MotifParams,
    Params,
    TraversalParams,
)


def _by_motif(index: GraphIndex, params: Params) -> dict[str, list]:
    out: dict[str, list] = {}
    for candidate in enumerate_candidates(index, params):
        out.setdefault(candidate.motif, []).append(candidate)
    return out


def test_all_four_motifs_fire_on_the_fixture(index: GraphIndex, params: Params) -> None:
    found = _by_motif(index, params)
    assert set(found) == {
        "gap_closure",
        "transitive_chain",
        "analogical_transfer",
        "condition_split",
    }


def test_gap_closure_covers_every_stated_gap(index: GraphIndex, params: Params) -> None:
    gaps = {c.gap_id for c in _by_motif(index, params)["gap_closure"]}
    assert gaps == {"g1", "g2"}


def test_gap_closure_carries_an_evidence_spine(index: GraphIndex, params: Params) -> None:
    """A gap the graph already almost connects is a far better proposal than a
    pair with no path at all, so the path is attached, not just the note."""
    g1 = next(c for c in _by_motif(index, params)["gap_closure"] if c.gap_id == "g1")
    assert [e.link_id for e in g1.path] == ["L5", "L6", "L7"]


def test_chains_never_restate_an_existing_link(index: GraphIndex, params: Params) -> None:
    for candidate in _by_motif(index, params)["transitive_chain"]:
        assert not index.links_between(candidate.subject, candidate.object)


def test_chains_are_at_least_two_hops(index: GraphIndex, params: Params) -> None:
    """One hop is a stated fact. Composing it is not a hypothesis."""
    assert all(c.hops >= 2 for c in _by_motif(index, params)["transitive_chain"])


def test_repurposing_shape_is_tagged(index: GraphIndex, params: Params) -> None:
    tagged = [
        c
        for c in _by_motif(index, params)["transitive_chain"]
        if "repurposing" in c.tags
    ]
    assert tagged
    for candidate in tagged:
        assert index.kind(candidate.object) == "disease"


def test_reversed_chains_are_flagged(index: GraphIndex, params: Params) -> None:
    reversed_chains = [
        c for c in enumerate_candidates(index, params) if "reversed_edge" in c.tags
    ]
    assert reversed_chains
    for candidate in reversed_chains:
        assert any(not e.forward for e in candidate.path)


def test_analogy_proposes_in_both_directions(index: GraphIndex) -> None:
    """The interesting transfer is whichever thing has the edge the other
    lacks, and which one that is has nothing to do with id order."""
    params = Params(motifs=MotifParams(enabled=("analogical_transfer",)))
    analogies = enumerate_candidates(index, params)
    receivers = {c.subject for c in analogies}
    donors = {c.analogues[0] for c in analogies}
    assert "t1" in receivers and "t2" in donors


def test_analogy_needs_more_than_a_shared_hub(index: GraphIndex) -> None:
    strict = Params(
        motifs=MotifParams(enabled=("analogical_transfer",), analogy_min_jaccard=0.99)
    )
    assert enumerate_candidates(index, strict) == []


def test_analogy_stays_within_a_kind_by_default(index: GraphIndex) -> None:
    params = Params(motifs=MotifParams(enabled=("analogical_transfer",)))
    for candidate in enumerate_candidates(index, params):
        donor = candidate.analogues[0]
        assert index.kind(donor) == index.kind(candidate.subject)


def test_condition_split_only_on_disagreement(index: GraphIndex, params: Params) -> None:
    splits = _by_motif(index, params)["condition_split"]
    assert [c.focus_link_id for c in splits] == ["L6"]
    assert index.links["L6"].state == "disagreed"
    assert len(splits[0].conditions) == 2


def test_condition_split_requires_a_named_condition(index: GraphIndex) -> None:
    """Without two stated `where` values the hypothesis is "maybe it's
    conditions" with no candidate condition in hand."""
    clone = index.graph.model_copy(deep=True)
    for finding in clone.findings:
        finding.where = None
    stripped = GraphIndex(clone)
    params = Params(motifs=MotifParams(enabled=("condition_split",)))
    assert enumerate_candidates(stripped, params) == []
    lenient = Params(
        motifs=MotifParams(
            enabled=("condition_split",), condition_split_requires_where=False
        )
    )
    assert enumerate_candidates(stripped, lenient)


def test_framing_anchors_restrict_the_seed_side(index: GraphIndex) -> None:
    params = Params(framing=FramingParams(anchors=("metformin",)))
    for candidate in enumerate_candidates(index, params):
        if candidate.motif == "transitive_chain":
            assert candidate.subject == "t8"


def test_framing_accepts_names_and_aliases(index: GraphIndex) -> None:
    by_alias = Params(framing=FramingParams(anchors=("IPF",)))
    by_id = Params(framing=FramingParams(anchors=("t5",)))
    assert [c.key() for c in enumerate_candidates(index, by_alias)] == [
        c.key() for c in enumerate_candidates(index, by_id)
    ]


def test_exclusions_route_around_a_node(index: GraphIndex) -> None:
    """The hub a clinician already knows about should not appear at all --
    interior included, which the per-edge gate cannot see."""
    params = Params(framing=FramingParams(exclude=("TGF-beta1",)))
    for candidate in enumerate_candidates(index, params):
        assert "t3" not in candidate.node_ids()


def test_closed_mode_fixes_both_ends(index: GraphIndex) -> None:
    params = Params(
        framing=FramingParams(mode="closed", anchors=("t8",), targets=("t5",)),
        motifs=MotifParams(enabled=("transitive_chain",)),
    )
    candidates = enumerate_candidates(index, params)
    assert candidates
    for candidate in candidates:
        assert (candidate.subject, candidate.object) == ("t8", "t5")


def test_kind_filters_produce_the_repurposing_shape(index: GraphIndex) -> None:
    params = Params(
        traversal=TraversalParams(
            max_hops=4, seed_kinds=("small_molecule",), target_kinds=("disease",)
        ),
        motifs=MotifParams(enabled=("transitive_chain",)),
    )
    candidates = enumerate_candidates(index, params)
    assert candidates
    for candidate in candidates:
        assert index.kind(candidate.subject) == "small_molecule"
        assert index.kind(candidate.object) == "disease"


def test_mirror_images_are_deduped(index: GraphIndex, params: Params) -> None:
    """BFS from every seed finds A→B and B→A over the same links. They are one
    hypothesis stated from two ends."""
    keys = [c.key() for c in enumerate_candidates(index, params)]
    assert len(keys) == len(set(keys))


def test_enumeration_respects_the_cap(index: GraphIndex) -> None:
    params = Params(traversal=TraversalParams(max_candidates=3))
    assert len(enumerate_candidates(index, params)) <= 3


def test_condition_splits_respect_exclusions(index: GraphIndex) -> None:
    """A split is about a link, not a path, so the seed/target gates miss it --
    the exclusion has to be applied to the link's own endpoints."""
    params = Params(
        framing=FramingParams(exclude=("AMPK",)),
        motifs=MotifParams(enabled=("condition_split",)),
    )
    assert enumerate_candidates(index, params) == []


def test_condition_splits_stay_in_the_framing_scope(index: GraphIndex) -> None:
    in_scope = Params(
        framing=FramingParams(anchors=("metformin",)),
        motifs=MotifParams(enabled=("condition_split",)),
    )
    out_of_scope = Params(
        framing=FramingParams(anchors=("PDGFR",)),
        traversal=TraversalParams(max_hops=1),
        motifs=MotifParams(enabled=("condition_split",)),
    )
    assert enumerate_candidates(index, in_scope)
    assert enumerate_candidates(index, out_of_scope) == []


def test_focus_is_applied_before_the_global_candidate_cap(index: GraphIndex) -> None:
    params = Params(traversal=TraversalParams(max_candidates=1))
    focused = enumerate_candidates(index, params, focus_thing_id="t8")
    assert focused
    assert all("t8" in candidate.node_ids() for candidate in focused)
