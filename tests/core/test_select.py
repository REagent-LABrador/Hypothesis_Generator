"""Selection: thresholds, the Pareto front, MMR, and the quotas."""

from __future__ import annotations

from hyp_gen.generate.candidates import Candidate, enumerate_candidates
from hyp_gen.graph import GraphIndex
from hyp_gen.params import Params, SelectionParams
from hyp_gen.generate.scoring import score_all
from hyp_gen.generate.select import pareto_front, select, similarity


def _scored(index: GraphIndex, params: Params):
    return score_all(index, enumerate_candidates(index, params), params)


def test_top_k_is_respected(index: GraphIndex) -> None:
    params = Params(selection=SelectionParams(top_k=3))
    assert len(select(_scored(index, params), params)) == 3


def test_thresholds_drop_the_unworthy(index: GraphIndex) -> None:
    params = Params(selection=SelectionParams(min_support=0.99))
    assert select(_scored(index, params), params) == []


def test_contradiction_ceiling_filters(index: GraphIndex) -> None:
    strict = Params(selection=SelectionParams(max_contradiction_risk=0.0))
    for _, scores in select(_scored(index, strict), strict):
        assert scores.contradiction_risk == 0.0


def test_pareto_front_keeps_only_non_dominated(index: GraphIndex, params: Params) -> None:
    scored = _scored(index, params)
    front = pareto_front(scored)
    assert front
    for _, candidate_scores in front:
        # Nothing in the full set may beat a front member on every axis.
        assert not any(
            other.support >= candidate_scores.support
            and other.novelty >= candidate_scores.novelty
            and other.testability >= candidate_scores.testability
            and (
                other.support > candidate_scores.support
                or other.novelty > candidate_scores.novelty
                or other.testability > candidate_scores.testability
            )
            for _, other in scored
        )


def test_mmr_reduces_redundancy(index: GraphIndex) -> None:
    """Pure score returns several versions of one idea; that is the failure MMR
    exists to fix, so the diverse record must overlap less."""
    greedy = Params(selection=SelectionParams(top_k=5, diversity_lambda=1.0))
    diverse = Params(selection=SelectionParams(top_k=5, diversity_lambda=0.3))

    def overlap(chosen: list[tuple[Candidate, object]]) -> float:
        picks = [c for c, _ in chosen]
        pairs = [
            similarity(a, b, "jaccard_nodes")
            for i, a in enumerate(picks)
            for b in picks[i + 1 :]
        ]
        return sum(pairs) / len(pairs) if pairs else 0.0

    assert overlap(select(_scored(index, diverse), diverse)) <= overlap(
        select(_scored(index, greedy), greedy)
    )


def test_lambda_one_is_plain_rank_order(index: GraphIndex) -> None:
    params = Params(
        selection=SelectionParams(
            top_k=4, diversity_lambda=1.0, max_per_subject=0, max_per_object=0
        )
    )
    scored = _scored(index, params)
    chosen = select(scored, params)
    assert [c.id for c, _ in chosen] == [c.id for c, _ in scored[:4]]


def test_subject_quota_caps_one_fashionable_target(index: GraphIndex) -> None:
    params = Params(selection=SelectionParams(top_k=8, max_per_subject=1))
    subjects = [c.subject for c, _ in select(_scored(index, params), params)]
    assert len(subjects) == len(set(subjects))


def test_motif_quota_forces_a_mixed_slate(index: GraphIndex) -> None:
    params = Params(selection=SelectionParams(top_k=8, max_per_motif=1))
    motifs = [c.motif for c, _ in select(_scored(index, params), params)]
    assert len(motifs) == len(set(motifs))


def test_quota_block_frees_the_slot_for_a_different_idea(index: GraphIndex) -> None:
    """A rejected pick must not shrink the record -- the next-best *different*
    candidate should take the slot."""
    unquota = Params(selection=SelectionParams(top_k=5, max_per_subject=0))
    quota = Params(selection=SelectionParams(top_k=5, max_per_subject=1))
    assert len(select(_scored(index, quota), quota)) == len(
        select(_scored(index, unquota), unquota)
    )


def test_similarity_modes_differ(index: GraphIndex, params: Params) -> None:
    a, b = [c for c, _ in _scored(index, params)][:2]
    assert 0.0 <= similarity(a, b, "jaccard_nodes") <= 1.0
    assert similarity(a, a, "endpoint") == 1.0
    assert similarity(a, a, "motif") == 1.0


def test_selection_is_deterministic(index: GraphIndex, params: Params) -> None:
    first = [c.id for c, _ in select(_scored(index, params), params)]
    second = [c.id for c, _ in select(_scored(index, params), params)]
    assert first == second
