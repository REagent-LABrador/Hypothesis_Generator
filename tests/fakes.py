"""Test doubles shared by the core and adapter suites.

``FakeJudge`` is why this repo has model-stage coverage without a network call
or a key: it answers with schema-valid objects, and can be told to refuse, to
run out of budget, or to cite evidence it was never shown.
"""

from __future__ import annotations

from hyp_gen.hypothesis import Articulation, Claim, Comparison, Critique
from hyp_gen.params import Params, RankingParams, SelectionParams
from hyp_gen.reasoning.llm import BudgetExceeded


class FakeJudge:
    """Returns schema-valid answers without touching the network."""

    def __init__(self, *, cite: str | None = None, verdict: str = "partly_supported",
                 raises: Exception | None = None, max_calls: int = 40,
                 prefers: str | None = None) -> None:
        self.cite = cite
        self.verdict = verdict
        self.raises = raises
        self.max_calls = max_calls
        # `prefers` makes comparisons content-based rather than positional:
        # whichever side's evidence pack mentions this id wins, wherever it is
        # shown. Without it the fake always answers "A", which is the position
        # bias the tournament is supposed to detect.
        self.prefers = prefers
        self.calls = 0
        self.systems: list[str] = []

    def parse(self, *, system, prompt, schema, effort="high", max_tokens=8000):
        self.calls += 1
        if self.calls > self.max_calls:
            raise BudgetExceeded("out of budget")
        if self.raises is not None:
            raise self.raises
        self.systems.append(system)
        if schema is Articulation:
            cites = [self.cite] if self.cite else []
            return Articulation(
                statement="A causes B under condition C.",
                mechanism="A -> X -> B",
                claims=[Claim(text="A binds X", cites=cites, inferred=not cites)],
                novel_because="the graph never states A to B",
                predictions=["X rises before B"],
                falsifier="B occurs with X knocked out",
                decisive_experiment="knock out X and measure B",
                assumptions=["X is measurable in this system"],
            )
        if schema is Critique:
            return Critique(
                verdict=self.verdict,
                strongest_objection="the middle link is single-source",
                unsupported_leaps=["A to B is not stated"],
            )
        if schema is Comparison:
            winner = "A"
            if self.prefers:
                first, _, second = prompt.partition("=" * 60)
                if self.prefers not in first and self.prefers in second:
                    winner = "B"
            return Comparison(winner=winner, margin="clear", reason="better evidence")
        raise AssertionError(f"unexpected schema {schema}")


def _params(**ranking) -> Params:
    return Params(
        selection=SelectionParams(top_k=3),
        ranking=RankingParams(**ranking),
    )
