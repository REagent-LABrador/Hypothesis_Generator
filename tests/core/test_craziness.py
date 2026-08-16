"""The craziness dial: one float from super-safe to very ambitious.

The tests that matter here are not the interpolation arithmetic -- that is a
lerp and it either works or it does not. They are the two boundaries: that the
dial reaches far enough at 1.0 to produce the kind of hypothesis it exists for,
and that it cannot reach the things it must never move.
"""

from __future__ import annotations

from collections import Counter

import pytest

from hyp_gen.generate.candidates import enumerate_candidates
from hyp_gen.graph import GraphIndex, KnowledgeGraph
from hyp_gen.params import CRAZINESS_NEVER_TOUCHES, PROFILES, Params
from hyp_gen.pipeline import Generator
from hyp_gen.generate.scoring import score_all
from hyp_gen.generate.select import select

DIAL = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)


def _resolve(params: Params, path: str):
    value = params
    for name in path.split("."):
        value = value[name] if isinstance(value, dict) else getattr(value, name)
    return value


def _selected(index: GraphIndex, params: Params):
    return select(score_all(index, enumerate_candidates(index, params), params), params)


# -- the reason the dial exists --------------------------------------------


def test_high_craziness_surfaces_the_different_field_hypothesis(index: GraphIndex) -> None:
    """"I read this in a slightly different field, maybe it works here" is an
    analogical transfer, and it is the whole point of the top of the dial.

    This is a regression test with a story. The obvious way to build this dial
    is to raise `selection.min_novelty` with craziness, and doing so removes
    every one of the 90 analogical transfers this graph enumerates at 1.0 --
    because novelty is measured as distance and an analogy is one hop. The
    ambitious setting returned twelve long chains and not one leap.
    """
    motifs = Counter(c.motif for c, _ in _selected(index, Params.at_craziness(1.0)))
    assert motifs["analogical_transfer"] >= 1

    safe = Counter(c.motif for c, _ in _selected(index, Params.at_craziness(0.0)))
    assert safe["analogical_transfer"] == 0


def test_the_dial_actually_changes_the_slate(graph: KnowledgeGraph) -> None:
    safe = Generator(graph=graph, params=Params.at_craziness(0.0)).run()
    wild = Generator(graph=graph, params=Params.at_craziness(1.0)).run()

    assert len(safe.hypotheses) < len(wild.hypotheses)
    assert max(h.hops for h in wild.hypotheses) > max(h.hops for h in safe.hypotheses)
    assert {h.id for h in safe.hypotheses} != {h.id for h in wild.hypotheses}


def test_a_safe_run_stays_near_what_is_already_known(graph: KnowledgeGraph) -> None:
    safe = Generator(graph=graph, params=Params.at_craziness(0.0)).run()
    assert all(h.hops <= 2 for h in safe.hypotheses)
    assert all(h.scores["support"] >= 0.4 for h in safe.hypotheses)


# -- what the dial must never move -----------------------------------------


@pytest.mark.parametrize("path", sorted(CRAZINESS_NEVER_TOUCHES))
def test_forbidden_fields_are_identical_across_the_whole_dial(path: str) -> None:
    """Craziness changes what you will propose, never what the evidence says.

    If the support arithmetic moved with ambition, the dial would be a licence
    to launder a weak chain into a strong one -- and the score would stop
    meaning anything across two runs.
    """
    values = {repr(_resolve(Params.at_craziness(c), path)) for c in DIAL}
    assert len(values) == 1, f"{path} moved with craziness: {values}"


def test_the_same_evidence_scores_the_same_at_any_craziness(index: GraphIndex) -> None:
    """The rule above, measured rather than asserted.

    Keyed on the actual chain of links, not on the candidate id: ids are not
    unique across traversal settings -- `H-chain-t3-t5-2` names three different
    two-hop paths between the same endpoints, and a wider beam finds more of
    them. Which paths exist is the dial's business. What a given path is worth
    is not.

    One evidence-side knob does move: `min_independent_groups`, which caps a
    link resting on a single research group. That is a *standard* rather than a
    weight -- how much corroboration this run requires -- so it is excluded
    here and the assertion runs above the cliff at 0.25.
    """
    by_path: dict[tuple, set[float]] = {}
    for craziness in (c for c in DIAL if c >= 0.25):
        params = Params.at_craziness(craziness)
        for candidate, scores in score_all(index, enumerate_candidates(index, params), params):
            key = (candidate.motif, candidate.link_ids)
            by_path.setdefault(key, set()).add(round(scores.support, 9))

    moved = {key: seen for key, seen in by_path.items() if len(seen) > 1}
    assert not moved, f"support moved with craziness for {list(moved)[:3]}"


def test_the_corroboration_cap_is_visible_when_it_bites(index: GraphIndex) -> None:
    """The one evidence-side effect the dial has, and it announces itself."""
    params = Params.at_craziness(0.0)
    notes = [
        note
        for _, scores in score_all(index, enumerate_candidates(index, params), params)
        for note in scores.notes
    ]
    assert any("single research group" in note for note in notes)


def test_untrustworthy_output_always_halts_verification() -> None:
    """`structure` and `citations` mean the output cannot be believed, which is
    orthogonal to how ambitious it was being. Only `independence` -- how much
    corroboration you require -- is allowed to move."""
    for craziness in DIAL:
        halting = Params.at_craziness(craziness).verification.halt_on
        assert "structure" in halting
        assert "citations" in halting

    assert "independence" in Params.at_craziness(0.0).verification.halt_on
    assert "independence" not in Params.at_craziness(1.0).verification.halt_on


def test_broken_inference_forms_stay_off_at_every_level() -> None:
    """Two correlations in a row imply nothing at 0.0 and imply nothing at 1.0.
    Chaining them is not ambition, it is a broken inference."""
    for craziness in (0.0, 1.0):
        traversal = Params.at_craziness(craziness).traversal
        assert "correlates_with" in traversal.predicates_deny
        assert traversal.allow_no_effect_edges is False
        assert traversal.allow_negative_edges is False


def test_scrutiny_rises_with_craziness_rather_than_falling() -> None:
    safe = Params.at_craziness(0.0).ranking
    wild = Params.at_craziness(1.0).ranking
    assert wild.critics_per_hypothesis >= safe.critics_per_hypothesis
    assert wild.evolution_rounds >= safe.evolution_rounds


# -- the shape of the schedule ---------------------------------------------


def test_bounds_are_enforced() -> None:
    for bad in (-0.01, 1.01, 2.0):
        with pytest.raises(ValueError, match="between 0 and 1"):
            Params.at_craziness(bad)
    Params.at_craziness(0.0)
    Params.at_craziness(1.0)


def test_the_aperture_opens_monotonically() -> None:
    hops = [Params.at_craziness(c).traversal.max_hops for c in DIAL]
    confidence = [Params.at_craziness(c).traversal.min_link_confidence for c in DIAL]
    jaccard = [Params.at_craziness(c).motifs.analogy_min_jaccard for c in DIAL]

    assert hops == sorted(hops)
    assert confidence == sorted(confidence, reverse=True)
    assert jaccard == sorted(jaccard, reverse=True)


def test_hub_damping_rises_again_at_the_top() -> None:
    """The one knob that is not monotonic, and deliberately so: extra hops are
    only worth having if they are not all routed through one promiscuous node.
    Reaching further and reaching through a hub are different things."""
    damping = [Params.at_craziness(c).traversal.hub_damping for c in DIAL]
    assert min(damping) == Params.at_craziness(0.5).traversal.hub_damping
    assert damping[-1] > min(damping)


def test_anchors_reproduce_the_named_profiles() -> None:
    safe, middle = Params.at_craziness(0.0), Params.at_craziness(0.5)
    conservative, default = PROFILES["conservative"], PROFILES["default"]

    assert safe.traversal.max_hops == conservative.traversal.max_hops
    assert safe.traversal.min_link_confidence == conservative.traversal.min_link_confidence
    assert safe.selection.top_k == conservative.selection.top_k
    assert safe.evidence.min_independent_groups == conservative.evidence.min_independent_groups

    assert middle.traversal.max_hops == default.traversal.max_hops
    assert middle.selection.top_k == default.selection.top_k
    assert middle.novelty.popularity_penalty == default.novelty.popularity_penalty


def test_cross_kind_analogy_is_the_last_guard_to_come_off() -> None:
    assert Params.at_craziness(0.5).motifs.analogy_same_kind_only is True
    assert Params.at_craziness(1.0).motifs.analogy_same_kind_only is False


# -- composition -----------------------------------------------------------


def test_the_profile_keeps_the_shape_and_craziness_takes_the_appetite() -> None:
    """A profile says what question to ask; craziness says how far to reach."""
    for craziness in (0.0, 1.0):
        params = Params.at_craziness(craziness, "valuation")
        assert params.traversal.seed_kinds == ("small_molecule",)
        assert params.traversal.target_kinds == ("disease",)
        assert params.selection.max_per_subject == 2

    assert Params.at_craziness(1.0, "valuation").traversal.max_hops > Params.at_craziness(
        0.0, "valuation"
    ).traversal.max_hops


def test_craziness_may_narrow_the_motifs_but_never_widen_them() -> None:
    """`mechanism` deliberately excludes analogical transfer. Turning the dial
    up must not hand it back a motif the profile ruled out."""
    enabled = Params.at_craziness(1.0, "mechanism").motifs.enabled
    assert "analogical_transfer" not in enabled
    assert set(enabled) <= set(PROFILES["mechanism"].motifs.enabled)


def test_explicit_overrides_beat_the_dial() -> None:
    params = Params.at_craziness(1.0, overrides={"traversal": {"max_hops": 2}})
    assert params.traversal.max_hops == 2
    assert params.motifs.analogy_same_kind_only is False  # the rest of the dial stands


def test_the_stance_is_recorded_so_a_slate_can_say_where_it_came_from() -> None:
    params = Params.at_craziness(0.7, "repurposing")
    assert params.stance.craziness == 0.7
    assert params.stance.profile == "repurposing"
    assert Params.profile("conservative").stance.profile == "conservative"


def test_the_same_dial_setting_is_the_same_run(graph: KnowledgeGraph) -> None:
    """A hypothesis is a function of (graph, params) and nothing else, and
    craziness is a way of writing params -- not an escape from that."""
    first = Generator(graph=graph, params=Params.at_craziness(0.63)).run()
    second = Generator(graph=graph, params=Params.at_craziness(0.63)).run()
    assert [h.id for h in first.hypotheses] == [h.id for h in second.hypotheses]
    assert [h.rank_score for h in first.hypotheses] == [h.rank_score for h in second.hypotheses]
