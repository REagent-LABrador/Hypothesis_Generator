"""End to end, including the model stages driven by a scripted fake judge.

The fake judge is not a mock of convenience -- it is how the harness's own
failure handling gets tested. A judge that refuses, that runs out of budget, or
that cites something it was never shown are all things the real one will
eventually do, and all three must degrade into a visible warning rather than a
crash or a silent pass.
"""

from __future__ import annotations

import json

import pytest

from hyp_gen.generate.evidence import build_pack
from hyp_gen.graph import GraphIndex, KnowledgeGraph
from hyp_gen.reasoning.llm import BudgetExceeded, RefusalError
from hyp_gen.params import (
    BudgetParams,
    EvidenceParams,
    LoopParams,
    MotifParams,
    Params,
    RankingParams,
    SelectionParams,
)
from hyp_gen.pipeline import Generator
from adapters.report import render as report
from adapters.report.render import to_markdown
from hyp_gen.hypothesis import (
    Articulation,
    Claim,
    Comparison,
    Critique,
    HypothesisDocument,
)
from hyp_gen.pipeline import RunResult
from hyp_gen.generate.scoring import score_candidate
from fakes import FakeJudge, _params


def test_runs_without_a_judge(graph: KnowledgeGraph, params: Params) -> None:
    """The deterministic half must stand alone -- this is what --dry-run and a
    keyless demo depend on."""
    record = Generator(graph=graph, params=params).run()
    assert isinstance(record, RunResult)
    assert record.hypotheses
    assert record.provenance.counts["model_calls"] == 0
    assert all(h.articulation is None for h in record.hypotheses)


def test_every_hypothesis_is_traceable(graph: KnowledgeGraph, params: Params) -> None:
    """Each one has to carry its own audit trail: links, findings, quotes."""
    record = Generator(graph=graph, params=params).run()
    index = GraphIndex(graph)
    for hypothesis in record.hypotheses:
        assert hypothesis.provenance
        for step in hypothesis.path:
            assert step["link"] in index.links
        for fid, finding in hypothesis.evidence["findings"].items():
            assert fid in index.findings
            assert finding["quote"], "a finding with no verbatim sentence is unciteable"


def test_articulates_and_critiques_with_lenses(graph: KnowledgeGraph) -> None:
    judge = FakeJudge()
    params = _params(critics_per_hypothesis=2)
    record = Generator(graph=graph, params=params, judge=judge).run()
    live = [h for h in record.hypotheses if not h.blocked]
    assert live
    for hypothesis in live:
        assert hypothesis.articulation is not None
        assert [c.lens for c in hypothesis.critiques] == ["mechanism", "evidence"]
        assert hypothesis.verdict is not None


def test_critic_count_drives_the_call_budget(graph: KnowledgeGraph) -> None:
    one = FakeJudge()
    three = FakeJudge()
    Generator(graph=graph, params=_params(critics_per_hypothesis=1), judge=one).run()
    Generator(graph=graph, params=_params(critics_per_hypothesis=3), judge=three).run()
    assert three.calls > one.calls


def test_consensus_needs_a_majority_to_refute(graph: KnowledgeGraph) -> None:
    """One lens calling it unsupported is information, not a ruling."""
    params = _params(critics_per_hypothesis=3, refute_threshold=0.9)
    record = Generator(graph=graph, params=params, judge=FakeJudge(verdict="unsupported")).run()
    live = [h for h in record.hypotheses if not h.blocked]
    assert all(h.verdict == "unsupported" for h in live)

    lenient = _params(critics_per_hypothesis=3, refute_threshold=0.34)
    slate2 = Generator(
        graph=graph, params=lenient, judge=FakeJudge(verdict="supported")
    ).run()
    assert all(h.verdict == "supported" for h in slate2.hypotheses if not h.blocked)


def test_illegal_citations_surface_as_errors(graph: KnowledgeGraph) -> None:
    judge = FakeJudge(cite="L-does-not-exist")
    record = Generator(graph=graph, params=_params(), judge=judge).run()
    codes = {i.code for h in record.hypotheses for i in h.issues}
    assert "illegal_citation" in codes


def test_a_refusal_degrades_to_a_warning(graph: KnowledgeGraph) -> None:
    judge = FakeJudge(raises=RefusalError("classifier declined"))
    record = Generator(graph=graph, params=_params(), judge=judge).run()
    assert record.hypotheses  # the run survives
    codes = {i.code for h in record.hypotheses for i in h.issues}
    assert "articulate_failed" in codes


def test_budget_stops_the_run_not_just_one_hypothesis(graph: KnowledgeGraph) -> None:
    """The ceiling is a stop. Retrying it once per survivor would burn the rest
    of the run rediscovering the same thing."""
    judge = FakeJudge(max_calls=1)
    params = Params(
        selection=SelectionParams(top_k=3),
        ranking=RankingParams(critics_per_hypothesis=2),
        budget=BudgetParams(max_model_calls=1),
    )
    record = Generator(graph=graph, params=params, judge=judge).run()
    assert judge.calls == 2  # the call that trips it, and no more
    codes = {i.code for h in record.hypotheses for i in h.issues}
    assert "skipped_no_budget" in codes


def test_stale_gaps_are_dropped_before_selection(graph: KnowledgeGraph) -> None:
    """A gap whose pair now has a link was promoted (or is wrong). Proposing it
    is a restatement, and it must not take a slot from a real candidate."""
    clone = graph.model_copy(deep=True)
    clone.gaps[0].missing = ["t1", "t3"]  # g1 now names a pair L1 already links
    wide = Params(selection=SelectionParams(top_k=12))
    record = Generator(graph=clone, params=wide).run()
    assert "H-g1" not in {h.id for h in record.hypotheses}
    assert "H-g2" in {h.id for h in record.hypotheses}  # the untouched gap survives
    assert all(not h.blocked for h in record.hypotheses)


def test_structurally_invalid_candidates_never_reach_the_model(
    graph: KnowledgeGraph,
) -> None:
    """Belt and braces: if a bad candidate does reach assembly, it is blocked
    before a model call is spent on it."""
    clone = graph.model_copy(deep=True)
    clone.gaps[0].missing = ["t1", "t3"]
    lenient = Params(
        selection=SelectionParams(top_k=12),
        motifs=MotifParams(require_unstated=False),
        ranking=RankingParams(critics_per_hypothesis=1),
    )
    judge = FakeJudge()
    record = Generator(graph=clone, params=lenient, judge=judge).run()

    stale = next(h for h in record.hypotheses if h.id == "H-g1")
    assert stale.blocked
    assert stale.articulation is None
    assert "already_stated" in {i.code for i in stale.issues}


def test_tournament_ranks_on_content(graph: KnowledgeGraph) -> None:
    params = _params(tournament=True, critics_per_hypothesis=1)
    judge = FakeJudge(prefers="L8")  # decides by evidence, not by position
    record = Generator(graph=graph, params=params, judge=judge).run()

    rated = [h for h in record.hypotheses if h.elo is not None]
    assert len(rated) >= 2
    assert rated == sorted(rated, key=lambda h: -h.elo)
    assert len({h.elo for h in rated}) > 1  # the debates moved something
    # The winner is the one whose pack actually contains the preferred link.
    assert "L8" in rated[0].evidence["links"]


def test_tournament_refuses_to_rank_a_position_biased_judge(graph: KnowledgeGraph) -> None:
    """A judge that always says "A" is exactly what swapped passes exist to
    catch: the verdict flips with the order, so nothing separates."""
    params = _params(tournament=True, critics_per_hypothesis=1, debate_turns=2)
    record = Generator(graph=graph, params=params, judge=FakeJudge()).run()
    rated = [h for h in record.hypotheses if h.elo is not None]
    assert len(rated) >= 2
    assert len({h.elo for h in rated}) == 1  # split verdicts, no movement


def test_single_debate_turn_takes_the_judge_at_its_word(graph: KnowledgeGraph) -> None:
    """With one pass there is no swap to disagree with, so even a biased judge
    produces a ranking -- which is the argument for debate_turns >= 2."""
    params = _params(tournament=True, critics_per_hypothesis=1, debate_turns=1)
    record = Generator(graph=graph, params=params, judge=FakeJudge()).run()
    rated = [h for h in record.hypotheses if h.elo is not None]
    assert len({h.elo for h in rated}) > 1


def test_evolution_revises_and_rechecks(graph: KnowledgeGraph) -> None:
    params = _params(critics_per_hypothesis=1, evolution_rounds=1, evolve_top_n=1)
    record = Generator(graph=graph, params=params, judge=FakeJudge()).run()
    evolved = [h for h in record.hypotheses if h.evolved_from]
    assert evolved
    assert evolved[0].evolution_operator in params.ranking.evolve_operators


def test_asks_are_off_until_the_loop_is_enabled(graph: KnowledgeGraph) -> None:
    assert Generator(graph=graph, params=Params()).run().asks == []

    looped = Params(loop=LoopParams(enabled=True, max_requests=5))
    record = Generator(graph=graph, params=looped).run()
    assert record.asks
    for ask in record.asks:
        assert ask.graph_id == graph.graph_id
        assert ask.ask in {"expand_node", "resolve_link", "test_gap", "new_question"}
        assert ask.reason


def test_asks_only_target_unsearched_gaps(graph: KnowledgeGraph) -> None:
    """g2 was already searched; asking the graph builder to look again wastes a round."""
    looped = Params(loop=LoopParams(enabled=True, max_requests=10))
    record = Generator(graph=graph, params=looped).run()
    gap_targets = {a.target for a in record.asks if a.ask == "test_gap"}
    assert "g2" not in gap_targets


def test_asks_are_deduped_and_capped(graph: KnowledgeGraph) -> None:
    looped = Params(loop=LoopParams(enabled=True, max_requests=2))
    record = Generator(graph=graph, params=looped).run()
    assert len(record.asks) <= 2
    keys = [(a.ask, a.target) for a in record.asks]
    assert len(keys) == len(set(keys))


def test_the_document_serialises_and_round_trips(graph: KnowledgeGraph) -> None:
    """The document is what a consumer parses, so it is what has to survive JSON."""
    record = Generator(graph=graph, params=_params(), judge=FakeJudge()).run()
    document = record.top()
    payload = json.loads(document.model_dump_json())

    reloaded = HypothesisDocument.model_validate(payload)
    assert reloaded.hypothesis.id == document.hypothesis.id
    assert reloaded.provenance.graph_id == record.provenance.graph_id
    # The params that produced it travel with it, or the run is not reproducible.
    assert payload["provenance"]["params"]["traversal"]["max_hops"]


def test_run_is_reproducible(graph: KnowledgeGraph, params: Params) -> None:
    first = Generator(graph=graph, params=params).run()
    second = Generator(graph=graph, params=params).run()
    assert [h.id for h in first.hypotheses] == [h.id for h in second.hypotheses]
    assert [h.rank_score for h in first.hypotheses] == [
        h.rank_score for h in second.hypotheses
    ]


def test_output_cap_is_enforced(graph: KnowledgeGraph) -> None:
    params = Params(
        selection=SelectionParams(top_k=8), budget=BudgetParams(max_output_hypotheses=2)
    )
    assert len(Generator(graph=graph, params=params).run().hypotheses) == 2


def test_evidence_pack_bounds_the_model_world(graph: KnowledgeGraph, params: Params) -> None:
    """Whatever is not in the pack cannot be cited, so the pack must contain
    every id the candidate rests on and nothing else."""
    index = GraphIndex(graph)
    generator = Generator(graph=graph, params=params)
    candidate, scores = generator.shortlist()[0]
    pack = build_pack(index, candidate, score_candidate(index, candidate, params))
    legal = pack.legal_ids()
    for link_id in candidate.link_ids:
        assert link_id in legal
    rendered = pack.to_prompt()
    assert "FINDINGS" in rendered and "PAPERS" in rendered
    unrelated = set(index.links) - set(candidate.link_ids)
    assert not (unrelated & legal)


@pytest.mark.parametrize("profile", ["default", "conservative", "speculative", "repurposing"])
def test_profiles_all_produce_something(graph: KnowledgeGraph, profile: str) -> None:
    from hyp_gen.params import PROFILES

    record = Generator(graph=graph, params=PROFILES[profile]).run()
    assert record.hypotheses, f"{profile} produced an empty record"


def test_conservative_is_stricter_than_speculative(graph: KnowledgeGraph) -> None:
    from hyp_gen.params import PROFILES

    conservative = Generator(graph=graph, params=PROFILES["conservative"]).run()
    speculative = Generator(graph=graph, params=PROFILES["speculative"]).run()
    assert len(conservative.hypotheses) <= len(speculative.hypotheses)
    assert max(h.scores["novelty"] for h in speculative.hypotheses) >= max(
        h.scores["novelty"] for h in conservative.hypotheses
    )


def test_every_hypothesis_carries_a_verification(
    graph: KnowledgeGraph, params: Params
) -> None:
    """Including the ones the model half never reached. A hypothesis with no
    gate table is one a reader cannot tell was checked."""
    record = Generator(graph=graph, params=params).run()
    assert record.hypotheses
    for hypothesis in record.hypotheses:
        assert hypothesis.verification is not None
        assert hypothesis.verification.gates


def test_verification_counts_reach_the_provenance(graph: KnowledgeGraph) -> None:
    record = Generator(graph=graph, params=_params(), judge=FakeJudge()).run()
    tallied = sum(
        record.provenance.counts[f"verification_{v}"]
        for v in ("verified", "qualified", "unverified", "rejected")
    )
    assert tallied == len(record.hypotheses)


def test_a_failed_gate_is_published_not_deleted(graph: KnowledgeGraph) -> None:
    """The gates added by the staged process express themselves through the
    verdict, never by making a hypothesis vanish from the record -- a check
    whose failures are invisible reads as assurance."""
    strict = Params(
        selection=SelectionParams(top_k=3),
        ranking=RankingParams(critics_per_hypothesis=1),
        evidence=EvidenceParams(min_independent_groups=99),
    )
    record = Generator(graph=graph, params=strict, judge=FakeJudge()).run()

    assert record.hypotheses
    for hypothesis in record.hypotheses:
        assert hypothesis.verification.halted_at == "independence"
        assert hypothesis.verification.verdict == "unverified"
        assert not hypothesis.blocked, "a gate failure must not block the record"


def test_halting_stops_the_model_gate_from_spending_calls(
    graph: KnowledgeGraph,
) -> None:
    """The reason the deterministic gates run first: critics are the expensive
    part, and a hypothesis already rejected must not pay for them."""
    strict = Params(
        selection=SelectionParams(top_k=3),
        ranking=RankingParams(critics_per_hypothesis=3),
        evidence=EvidenceParams(min_independent_groups=99),
    )
    judge = FakeJudge()
    record = Generator(graph=graph, params=strict, judge=judge).run()

    articulated = sum(1 for h in record.hypotheses if h.articulation is not None)
    # Articulation happens before verification, so exactly one call each and
    # not one critic call more.
    assert judge.calls == articulated
    assert all(h.critiques == [] for h in record.hypotheses)

