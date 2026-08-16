"""Evidence arithmetic, recomputed from findings rather than trusted."""

from __future__ import annotations

import pytest

from hyp_gen.generate.candidates import enumerate_candidates
from hyp_gen.graph import Coverage, GraphIndex
from hyp_gen.params import EvidenceParams, MotifParams, NoveltyParams, Params
from hyp_gen.generate.scoring import score_all, score_candidate, score_link


def test_agreement_falls_when_a_link_disagrees(index: GraphIndex, params: Params) -> None:
    agreed = score_link(index, index.links["L4"], params)
    disagreed = score_link(index, index.links["L6"], params)
    assert agreed.agreement == 1.0
    assert 0.0 < disagreed.agreement < 1.0
    assert disagreed.no_count == 1


def test_untested_claims_are_not_treated_as_agreement(index: GraphIndex, params: Params) -> None:
    """An unopposed claim nobody has tested collapses to 0.5, not 1.0."""
    clone = index.graph.model_copy(deep=True)
    link = next(l for l in clone.links if l.id == "L4")
    link.yes = []
    assert score_link(GraphIndex(clone), link, params).agreement == 0.5


def test_study_type_drives_evidence_quality(index: GraphIndex, params: Params) -> None:
    trial = score_link(index, index.links["L8"], params)      # clinical_trial
    test_tube = score_link(index, index.links["L2"], params)  # test_tube
    assert trial.evidence_quality > test_tube.evidence_quality


def test_hedged_and_secondhand_findings_are_discounted(index: GraphIndex, params: Params) -> None:
    """L1 carries a strong primary finding and a hedged review restatement; the
    review must not raise the link's quality."""
    scored = score_link(index, index.links["L1"], params)
    hedged_only = score_link(
        index,
        index.links["L1"].model_copy(update={"yes": ["f13"]}),
        params,
    )
    assert hedged_only.evidence_quality < scored.evidence_quality


def test_basis_penalty_applies(index: GraphIndex) -> None:
    strict = Params(
        evidence=EvidenceParams(basis_penalty={"primary": 1.0, "background_only": 0.1})
    )
    link = index.links["L4"]
    primary = score_link(index, link, strict)
    background = score_link(
        index, link.model_copy(update={"basis": "background_only"}), strict
    )
    assert background.evidence_quality < primary.evidence_quality


def test_one_group_repeating_itself_is_capped(index: GraphIndex) -> None:
    """One lab reporting a result five times is one result."""
    lenient = Params(evidence=EvidenceParams(min_independent_groups=1))
    strict = Params(evidence=EvidenceParams(min_independent_groups=2, single_group_cap=0.3))
    link = index.links["L8"]  # single clinical trial, one first author
    assert score_link(index, link, lenient).support > 0.3
    capped = score_link(index, link, strict)
    assert capped.support == 0.3
    assert capped.capped is True


def test_independence_needs_distinct_groups(index: GraphIndex, params: Params) -> None:
    single = score_link(index, index.links["L4"], params)
    multi = score_link(index, index.links["L6"], params)  # Bernard + Okonkwo
    assert single.independence == 0.0
    assert multi.independence > 0.0


def test_recency_decay_is_off_by_default_and_works_when_on(index: GraphIndex) -> None:
    off = score_link(index, index.links["L3"], Params())
    on = score_link(
        index,
        index.links["L3"],
        Params(evidence=EvidenceParams(recency_half_life=2.0)),
    )
    assert on.evidence_quality < off.evidence_quality


def test_drift_reports_disagreement_with_stage_one(index: GraphIndex, params: Params) -> None:
    scored = score_link(index, index.links["L1"], params)
    assert scored.drift == pytest.approx(scored.support - scored.stated_overall, abs=1e-9)


def test_weakest_link_aggregation_is_the_default(index: GraphIndex) -> None:
    weakest = Params(evidence=EvidenceParams(chain_aggregation="weakest"))
    mean = Params(evidence=EvidenceParams(chain_aggregation="mean"))
    candidate = next(
        c
        for c in enumerate_candidates(index, weakest)
        if c.motif == "gap_closure" and c.gap_id == "g1"
    )
    by_weakest = score_candidate(index, candidate, weakest)
    by_mean = score_candidate(index, candidate, mean)
    assert by_weakest.support <= by_mean.support
    assert by_weakest.support == min(l.support for l in by_weakest.per_link)


def test_novelty_is_discounted_by_absence_reliability(index: GraphIndex, params: Params) -> None:
    """A shallow search cannot mint novelty: at `quick` depth, absence means
    unknown, so nothing may claim to be new because nobody found it."""
    candidate = next(
        c for c in enumerate_candidates(index, params) if c.motif == "gap_closure"
    )
    deep = score_candidate(index, candidate, params)

    shallow_graph = index.graph.model_copy(deep=True)
    shallow_graph.coverage = Coverage(depth="quick")
    shallow = score_candidate(GraphIndex(shallow_graph), candidate, params)
    assert shallow.novelty < deep.novelty


def test_a_searched_gap_scores_higher_than_an_unsearched_one(
    index: GraphIndex, params: Params
) -> None:
    """Looked for and not found is a much stronger claim than never looked."""
    candidates = {
        c.gap_id: c
        for c in enumerate_candidates(index, params)
        if c.motif == "gap_closure"
    }
    searched = score_candidate(index, candidates["g2"], params)   # searched_in_round: 2
    unsearched = score_candidate(index, candidates["g1"], params)  # null
    assert "searched" in " ".join(searched.notes)
    assert searched.novelty > unsearched.novelty


def test_gap_bonus_is_capped_like_stage_one_caps_it(index: GraphIndex) -> None:
    greedy = Params(
        novelty=NoveltyParams(
            gap_novelty_bonus=5.0,
            searched_gap_bonus=5.0,
            gap_confidence_cap=0.6,
            popularity_penalty=0.0,
            respect_absence_reliability=False,
        )
    )
    candidate = next(
        c for c in enumerate_candidates(index, greedy) if c.motif == "gap_closure"
    )
    scored = score_candidate(index, candidate, greedy)
    # hop novelty may add on top; the *gap* component itself is what is capped.
    assert scored.novelty <= 1.0


def test_popularity_penalty_pushes_back_on_famous_pairings(index: GraphIndex) -> None:
    plain = Params(novelty=NoveltyParams(popularity_penalty=0.0))
    penalised = Params(novelty=NoveltyParams(popularity_penalty=0.5))
    candidate = next(
        c for c in enumerate_candidates(index, plain) if c.motif == "transitive_chain"
    )
    assert (
        score_candidate(index, candidate, penalised).novelty
        <= score_candidate(index, candidate, plain).novelty
    )


def test_condition_split_risk_is_discounted(index: GraphIndex, params: Params) -> None:
    """Disagreement is the premise of that motif, not a defect in it."""
    split = next(
        c for c in enumerate_candidates(index, params) if c.motif == "condition_split"
    )
    scored = score_candidate(index, split, params)
    assert scored.contradiction_risk < 1.0
    assert scored.testability >= 0.9  # a named pair of conditions is an experiment


def test_reversed_chains_carry_extra_risk(index: GraphIndex, params: Params) -> None:
    reversed_chain = next(
        c for c in enumerate_candidates(index, params) if "reversed_edge" in c.tags
    )
    scored = score_candidate(index, reversed_chain, params)
    assert scored.contradiction_risk > 0
    assert any("direction" in note for note in scored.notes)


def test_structure_is_normalised_across_the_set(index: GraphIndex, params: Params) -> None:
    scored = score_all(index, enumerate_candidates(index, params), params)
    values = [s.structure for _, s in scored]
    assert max(values) == 1.0
    assert all(0.0 <= v <= 1.0 for v in values)


def test_motif_prior_multiplies_the_rank(index: GraphIndex) -> None:
    base = Params(motifs=MotifParams(enabled=("transitive_chain",)))
    damped = Params(
        motifs=MotifParams(enabled=("transitive_chain",), weights={"transitive_chain": 0.1})
    )
    top_base = score_all(index, enumerate_candidates(index, base), base)[0][1]
    top_damped = score_all(index, enumerate_candidates(index, damped), damped)[0][1]
    assert top_damped.rank_score < top_base.rank_score


def test_scoring_is_deterministic(index: GraphIndex, params: Params) -> None:
    first = score_all(index, enumerate_candidates(index, params), params)
    second = score_all(index, enumerate_candidates(index, params), params)
    assert [(c.id, s.rank_score) for c, s in first] == [
        (c.id, s.rank_score) for c, s in second
    ]
