"""The prompts themselves.

These are unglamorous but they are where the system's guarantees actually live:
if the evidence pack does not reach the prompt, or reaches it twice, or reaches
it without its caveats, every downstream check is validating a fiction.
"""

from __future__ import annotations

from hyp_gen.reasoning import reason
from hyp_gen.generate.candidates import enumerate_candidates
from hyp_gen.generate.evidence import build_pack
from hyp_gen.graph import GraphIndex
from hyp_gen.params import Params, RankingParams
from hyp_gen.hypothesis import Articulation, Claim, Comparison, Critique
from hyp_gen.generate.scoring import score_candidate


class RecordingJudge:
    """Captures what it was asked, returns the minimum valid answer."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.efforts: list[str] = []

    def parse(self, *, system, prompt, schema, effort="high", max_tokens=8000):
        self.systems.append(system)
        self.prompts.append(prompt)
        self.efforts.append(effort)
        if schema is Articulation:
            return Articulation(
                statement="s", mechanism="m", claims=[Claim(text="c")],
                novel_because="n", falsifier="f", decisive_experiment="d",
            )
        if schema is Critique:
            return Critique(verdict="partly_supported", strongest_objection="o")
        if schema is Comparison:
            return Comparison(winner="A", margin="clear", reason="r")
        raise AssertionError(schema)


def _packs(index: GraphIndex, params: Params, count: int = 2):
    out = []
    for candidate in enumerate_candidates(index, params)[:count]:
        out.append(build_pack(index, candidate, score_candidate(index, candidate, params)))
    return out


def test_articulation_prompt_carries_the_pack(index: GraphIndex, params: Params) -> None:
    pack = _packs(index, params, 1)[0]
    judge = RecordingJudge()
    reason.articulate(judge, pack, params)

    prompt = judge.prompts[0]
    for link_id in pack.links:
        assert link_id in prompt
    for finding in pack.findings.values():
        assert finding["quote"] in prompt, "the verbatim sentence must reach the model"
    for caveat in pack.caveats:
        assert caveat in prompt


def test_articulation_prompt_states_the_claim_budget(index: GraphIndex) -> None:
    params = Params(ranking=RankingParams(max_claims_per_hypothesis=3))
    judge = RecordingJudge()
    reason.articulate(judge, _packs(index, params, 1)[0], params)
    assert "at most 3" in judge.prompts[0]


def test_per_stage_effort_is_passed_through(index: GraphIndex) -> None:
    params = Params(
        ranking=RankingParams(effort_articulate="low", effort_critique="max")
    )
    pack = _packs(index, params, 1)[0]
    judge = RecordingJudge()
    articulation = reason.articulate(judge, pack, params)
    reason.critique(judge, pack, articulation, "mechanism", params)
    assert judge.efforts == ["low", "max"]


def test_each_critic_gets_its_own_lens(index: GraphIndex, params: Params) -> None:
    pack = _packs(index, params, 1)[0]
    judge = RecordingJudge()
    articulation = reason.articulate(judge, pack, params)
    for lens in ("mechanism", "evidence"):
        result = reason.critique(judge, pack, articulation, lens, params)
        assert result.lens == lens  # set by the harness, not chosen by the model
    assert reason.LENS_BRIEF["mechanism"] in judge.prompts[1]
    assert reason.LENS_BRIEF["evidence"] in judge.prompts[2]


def test_comparison_shows_each_side_exactly_once(index: GraphIndex, params: Params) -> None:
    """Regression: implicit literal concatenation binds tighter than `*`, so a
    bare divider once multiplied the whole A section sixty times."""
    pack_a, pack_b = _packs(index, params, 2)
    judge = RecordingJudge()
    art = Articulation(
        statement="s", mechanism="m", claims=[Claim(text="c")],
        novel_because="n", falsifier="f", decisive_experiment="d",
    )
    reason.compare(judge, pack_a, art, pack_b, art)

    prompt = judge.prompts[0]
    assert prompt.count("HYPOTHESIS A") == 1
    assert prompt.count("HYPOTHESIS B") == 1
    assert prompt.count("=" * 60) == 1
    # And the divider actually separates them, which the swapped-order pass
    # in the tournament depends on.
    first, _, second = prompt.partition("=" * 60)
    assert "HYPOTHESIS A" in first and "HYPOTHESIS B" in second
    assert len(prompt) < 4 * len(pack_a.to_prompt() + pack_b.to_prompt())


def test_evolution_prompt_carries_the_criticism(index: GraphIndex, params: Params) -> None:
    pack = _packs(index, params, 1)[0]
    judge = RecordingJudge()
    articulation = reason.articulate(judge, pack, params)
    critique = Critique(
        verdict="unsupported",
        strongest_objection="the middle link is one test tube result",
        unsupported_leaps=["A to B is not stated"],
        lens="evidence",
    )
    reason.evolve(judge, pack, articulation, [critique], "specialise")

    prompt = judge.prompts[-1]
    assert "the middle link is one test tube result" in prompt
    assert "A to B is not stated" in prompt
    assert reason.EVOLVE_OPERATORS["specialise"] in prompt


def test_consensus_needs_the_threshold(params: Params) -> None:
    supported = [Critique(verdict="supported", strongest_objection="")] * 3
    assert reason.consensus(supported, params) == "supported"

    mixed = [
        Critique(verdict="supported", strongest_objection=""),
        Critique(verdict="unsupported", strongest_objection=""),
    ]
    assert reason.consensus(mixed, params) == "unsupported"  # default 0.5

    lenient = Params(ranking=RankingParams(refute_threshold=0.9))
    assert reason.consensus(mixed, lenient) == "partly_supported"

    contradicted = [Critique(verdict="contradicted", strongest_objection="")]
    assert reason.consensus(contradicted, params) == "contradicted"
    assert reason.consensus([], params) is None
