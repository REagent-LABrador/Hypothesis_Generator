"""Wiring. Graph in, record out.

    enumerate -> score -> select (MMR) -> pack -> validate
              -> articulate -> verify (staged gates, critics inside)
              -> [tournament] -> [evolve] -> asks

Verification is one staged process rather than checks sprinkled through this
file: ``verify.py`` owns the order, the halt rule, and the record. What lives
here is only the part that needs a Judge -- the closure that runs the critic
lenses and absorbs budget and refusal errors so a gate never has to know they
exist.

Everything up to `articulate` is deterministic and runs with no API key, which
is what ``--dry-run`` exercises. Model calls happen only for the survivors of
selection, so cost scales with ``selection.top_k``, not with graph size, and
every call is counted against ``budget.max_model_calls``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField

from hyp_gen.checks import validate, verify
from hyp_gen.generate import asks as asks_mod
from hyp_gen.generate import select
from hyp_gen.generate.candidates import Candidate, enumerate_candidates
from hyp_gen.generate.evidence import EvidencePack, build_pack
from hyp_gen.graph import GraphIndex, KnowledgeGraph
from hyp_gen.params import Params
from hyp_gen.reasoning import reason
from hyp_gen.reasoning.llm import BudgetExceeded, Judge, RefusalError
from hyp_gen.hypothesis import (
    Ask,
    Critique,
    Hypothesis,
    HypothesisDocument,
    Provenance,
    ValidationIssue,
)
from hyp_gen.generate.scoring import Scores, score_all


class RunResult(BaseModel):
    """INTERNAL. Everything one run assembled, before the boundary narrows it.

    Not a contract and not written to disk. The app's output is one
    ``HypothesisDocument`` -- see ``hyp_gen/OUTPUT_SCHEMA.md`` -- and this is
    the set-shaped working state that has to exist for the winner to be
    knowable: you cannot rank without candidates to rank, and the tournament
    compares hypotheses against each other by construction.

    ``top()`` is the boundary. Everything after it deals in single hypotheses.
    """

    model_config = ConfigDict(extra="allow")

    provenance: Provenance
    hypotheses: list[Hypothesis] = PydanticField(default_factory=list)
    asks: list[Ask] = PydanticField(default_factory=list)

    def top(self) -> HypothesisDocument | None:
        """The one hypothesis that crosses the boundary, or None if there is none.

        None is a real answer: a graph that supports nothing should say so
        rather than promote the least bad row.
        """
        if not self.hypotheses:
            return None
        best = self.hypotheses[0]
        return HypothesisDocument(
            provenance=self.provenance,
            hypothesis=best,
            asks=[a for a in self.asks if a.for_hypothesis in (None, best.id)],
        )


@dataclass
class Generator:
    graph: KnowledgeGraph
    params: Params
    judge: Judge | None = None
    index: GraphIndex = field(init=False)

    slow_enumeration: float | None = field(default=None, init=False)

    _candidates: dict[str, Candidate] = field(default_factory=dict, init=False)
    """The structural candidate behind each hypothesis id. Verification checks
    the shape as well as the prose, and a Hypothesis flattens the path into
    dicts, so the candidate has to survive assembly rather than be rebuilt."""

    _out_of_budget: str | None = field(default=None, init=False)
    """Set the first time a stage hits ``budget.max_model_calls``. Once it is
    set, no further model work is attempted: the ceiling is a stop, not a
    per-hypothesis speed bump, and retrying it once per survivor would burn the
    rest of the run discovering the same thing."""

    def __post_init__(self) -> None:
        self.index = GraphIndex(self.graph)

    # -- deterministic half -------------------------------------------------

    def shortlist(self) -> list[tuple[Candidate, Scores]]:
        started = time.monotonic()
        candidates = enumerate_candidates(self.index, self.params)
        elapsed = time.monotonic() - started
        if elapsed > self.params.budget.max_enumeration_seconds:
            # Not fatal -- what enumeration produced is still valid. Recorded
            # so a slow graph is visible rather than mysterious.
            self.slow_enumeration = round(elapsed, 1)
        return select.select(score_all(self.index, candidates, self.params), self.params)

    def _assemble(
        self, candidate: Candidate, scores: Scores, position: int
    ) -> tuple[Hypothesis, EvidencePack]:
        index = self.index
        pack = build_pack(index, candidate, scores)
        issues = validate.check_structure(index, candidate, pack)

        path = [
            {
                "link": e.link_id,
                "from": e.src,
                "from_name": index.name(e.src),
                "how": e.how,
                "to": e.dst,
                "to_name": index.name(e.dst),
                "reversed": not e.forward,
                "state": index.links[e.link_id].state if e.link_id in index.links else None,
                "support": next(
                    (l.support for l in scores.per_link if l.link_id == e.link_id), None
                ),
            }
            for e in candidate.path
        ]

        hypothesis = Hypothesis(
            id=candidate.id,
            motif=candidate.motif,
            subject=candidate.subject,
            object=candidate.object,
            subject_name=index.name(candidate.subject),
            object_name=index.name(candidate.object),
            hops=candidate.hops,
            tags=list(candidate.tags),
            path=path,
            scores=scores.vector(),
            rank_score=scores.rank_score,
            evidence={
                "links": pack.links,
                "findings": pack.findings,
                "papers": pack.papers,
                "things": pack.things,
                "gap": pack.gap,
                "per_link_support": [vars(l) for l in scores.per_link],
                "scoring_notes": scores.notes,
            },
            caveats=pack.caveats,
            issues=issues,
            provenance=(
                f"{candidate.motif} over {self.graph.graph_id}@round{self.graph.round}"
                f" via {', '.join(candidate.link_ids) or 'no links'}"
            ),
        )
        hypothesis.asks = asks_mod.asks_for(
            index, candidate, scores, self.params, hypothesis.id, position
        )
        self._candidates[hypothesis.id] = candidate
        return hypothesis, pack

    # -- model half ---------------------------------------------------------

    def _critic_runner(
        self, hypothesis: Hypothesis, pack: EvidencePack
    ) -> verify.CriticRunner:
        """A closure the adversarial gate can call without knowing about money.

        Budget exhaustion and refusals are recorded as issues on the hypothesis
        and turned into a shorter list of critiques. The gate then sees either
        critiques or nothing, and an empty list becomes a skip -- which is the
        honest reading, since a critic that never ran did not approve anything.
        """

        def run() -> list[Critique]:
            judge, r = self.judge, self.params.ranking
            assert judge is not None
            critiques: list[Critique] = []
            lenses = list(r.critic_lenses)[: max(r.critics_per_hypothesis, 0)]
            for lens in lenses:
                try:
                    critiques.append(
                        reason.critique(judge, pack, hypothesis.articulation, lens, self.params)
                    )
                except (RefusalError, BudgetExceeded) as exc:
                    if isinstance(exc, BudgetExceeded):
                        self._out_of_budget = str(exc)
                    hypothesis.issues.append(
                        ValidationIssue(
                            code="critique_failed",
                            detail=f"{lens}: {exc}",
                            severity="warning",
                        )
                    )
                    break
            return critiques

        return run

    def _reason_over(self, hypothesis: Hypothesis, pack: EvidencePack) -> None:
        """Articulate, then run the staged verification. Mutates ``hypothesis``.

        Articulation happens here and nowhere else; every check on what it
        produced -- including the critics -- happens inside ``verify``, in one
        recorded order.
        """
        judge = self.judge
        assert judge is not None

        try:
            hypothesis.articulation = reason.articulate(judge, pack, self.params)
        except BudgetExceeded as exc:
            self._out_of_budget = str(exc)
            hypothesis.issues.append(
                ValidationIssue(code="articulate_failed", detail=str(exc), severity="warning")
            )
        except RefusalError as exc:
            hypothesis.issues.append(
                ValidationIssue(code="articulate_failed", detail=str(exc), severity="warning")
            )

        self._verify(hypothesis, pack)

    def _verify(self, hypothesis: Hypothesis, pack: EvidencePack) -> None:
        """Run the gates over whatever state the hypothesis is in.

        Called after articulation and again after every evolution round: a
        revision is a new claim set, so the previous verification describes a
        hypothesis that no longer exists.
        """
        if not self.params.verification.enabled:
            return
        candidate = self._candidates[hypothesis.id]
        context = verify.GateContext(
            index=self.index,
            candidate=candidate,
            pack=pack,
            params=self.params,
            articulation=hypothesis.articulation,
            critics=(
                self._critic_runner(hypothesis, pack)
                if self.judge is not None and hypothesis.articulation is not None
                else None
            ),
        )
        verify.apply(hypothesis, context)

    def _tournament(
        self, live: list[tuple[Hypothesis, EvidencePack]]
    ) -> None:
        """Pairwise debates with Elo, as in the co-scientist.

        Off by default: it costs O(k) extra calls for a ranking the scores
        already approximate. It earns its keep when the top few are genuinely
        close, because pairwise judgement is far more reliable than absolute
        scoring -- "which of these two" is a question a model can answer, "is
        this an 8 or a 9" is not.
        """
        judge, r = self.judge, self.params.ranking
        assert judge is not None
        rated = [(h, p) for h, p in live if h.articulation is not None]
        if len(rated) < 2:
            return
        for hypothesis, _ in rated:
            hypothesis.elo = r.elo_initial

        # Deterministic pairing: adjacent ranks first, since those are the
        # comparisons the scores are least sure about.
        pairs = [
            (a, b)
            for a, b in combinations(range(len(rated)), 2)
            if b - a <= max(r.tournament_matches_per_hypothesis // 2, 1)
        ]
        for i, j in pairs:
            hyp_a, pack_a = rated[i]
            hyp_b, pack_b = rated[j]

            # Judge the pair `debate_turns` times, alternating which hypothesis
            # is presented first. Pairwise LLM judges have a position bias, and
            # a verdict that flips when you swap the order is not a verdict --
            # this is what turns that bias into visible disagreement instead of
            # a confident ranking built on presentation order.
            wins_a = 0.0
            clear = True
            try:
                for turn in range(max(r.debate_turns, 1)):
                    swapped = turn % 2 == 1
                    if swapped:
                        result = reason.compare(
                            judge, pack_b, hyp_b.articulation, pack_a, hyp_a.articulation
                        )
                        a_won = result.winner == "B"
                    else:
                        result = reason.compare(
                            judge, pack_a, hyp_a.articulation, pack_b, hyp_b.articulation
                        )
                        a_won = result.winner == "A"
                    wins_a += 1.0 if a_won else 0.0
                    clear = clear and result.margin == "clear"
            except (RefusalError, BudgetExceeded) as exc:
                if isinstance(exc, BudgetExceeded):
                    self._out_of_budget = str(exc)
                hyp_a.issues.append(
                    ValidationIssue(
                        code="tournament_stopped", detail=str(exc), severity="warning"
                    )
                )
                return

            turns = max(r.debate_turns, 1)
            score_a = wins_a / turns
            expected_a = 1.0 / (1.0 + 10 ** ((hyp_b.elo - hyp_a.elo) / 400.0))
            # A narrow win moves less than a clear one, and a pair that split
            # across swapped orders moves least of all -- score_a lands on 0.5
            # and the update is driven only by the Elo difference.
            k = r.elo_k * (1.0 if clear else 0.5)
            hyp_a.elo = round(hyp_a.elo + k * (score_a - expected_a), 1)
            hyp_b.elo = round(hyp_b.elo - k * (score_a - expected_a), 1)

    def _evolve(self, live: list[tuple[Hypothesis, EvidencePack]]) -> None:
        """Refine the leaders against their own criticism, in place."""
        judge, r = self.judge, self.params.ranking
        assert judge is not None
        leaders = [
            (h, p) for h, p in live if h.articulation is not None and not h.blocked
        ][: r.evolve_top_n]

        for round_index in range(r.evolution_rounds):
            for hypothesis, pack in leaders:
                if not hypothesis.critiques:
                    continue
                operator = r.evolve_operators[round_index % len(r.evolve_operators)]
                try:
                    revised = reason.evolve(
                        judge, pack, hypothesis.articulation, hypothesis.critiques, operator
                    )
                except (RefusalError, BudgetExceeded) as exc:
                    hypothesis.issues.append(
                        ValidationIssue(
                            code="evolve_failed", detail=str(exc), severity="warning"
                        )
                    )
                    continue
                hypothesis.evolved_from = hypothesis.articulation.statement
                hypothesis.evolution_operator = operator
                hypothesis.articulation = revised
                # A revision is a new claim set, so the previous verification
                # describes a hypothesis that no longer exists -- including the
                # critiques, which attacked the pre-revision wording. Re-verify
                # from scratch rather than inherit a verdict earned by different
                # prose. This is why evolution costs critics again: whether the
                # revision actually answered the objection is the only question
                # an evolution round exists to settle.
                hypothesis.issues = [
                    i for i in hypothesis.issues if i.code != "illegal_citation"
                ]
                hypothesis.critiques = []
                hypothesis.verdict = None
                self._verify(hypothesis, pack)

    # -- entry point --------------------------------------------------------

    def run(self) -> RunResult:
        shortlist = self.shortlist()
        r = self.params.ranking
        assembled: list[tuple[Hypothesis, EvidencePack]] = []

        for position, (candidate, scores) in enumerate(shortlist, start=1):
            assembled.append(self._assemble(candidate, scores, position))

        live = [(h, p) for h, p in assembled if not h.blocked]

        if self.judge is not None and r.articulate:
            for hypothesis, pack in live:
                if self._out_of_budget:
                    hypothesis.issues.append(
                        ValidationIssue(
                            code="skipped_no_budget",
                            detail=(
                                "not articulated: the run hit "
                                f"budget.max_model_calls ({self.params.budget.max_model_calls})"
                            ),
                            severity="warning",
                        )
                    )
                    continue
                self._reason_over(hypothesis, pack)
            if r.tournament and not self._out_of_budget:
                self._tournament(live)
            if r.evolution_rounds and not self._out_of_budget:
                self._evolve(live)

        # Anything the model half never reached still gets the deterministic
        # gates. A blocked candidate, a budget casualty and a --dry-run
        # hypothesis are all worth a gate table: it says which checks passed
        # before the process stopped, which is exactly what "unverified" needs
        # to mean something.
        for hypothesis, pack in assembled:
            if hypothesis.verification is None:
                self._verify(hypothesis, pack)

        hypotheses = [h for h, _ in assembled][: self.params.budget.max_output_hypotheses]
        if r.tournament and any(h.elo is not None for h in hypotheses):
            hypotheses.sort(key=lambda h: (h.elo is None, -(h.elo or 0)))

        loop_asks = asks_mod.dedupe([a for h in hypotheses for a in h.asks])
        return RunResult(
            provenance=Provenance(
                graph_id=self.graph.graph_id,
                round=self.graph.round,
                question=self.graph.question,
                generated_at=self.graph.generated_at,
                params=self.params.model_dump(mode="json"),
                coverage=self.graph.coverage.model_dump(mode="json"),
                counts={
                    "things": len(self.graph.things),
                    "links": len(self.graph.links),
                    "findings": len(self.graph.findings),
                    "gaps": len(self.graph.gaps),
                    "shortlisted": len(shortlist),
                    "blocked": sum(1 for h in hypotheses if h.blocked),
                    "model_calls": self.judge.calls if self.judge else 0,
                    **{
                        f"verification_{verdict}": sum(
                            1
                            for h in hypotheses
                            if h.verification and h.verification.verdict == verdict
                        )
                        for verdict in ("verified", "qualified", "unverified", "rejected")
                    },
                },
                considered=len(hypotheses),
            ),
            hypotheses=hypotheses,
            asks=loop_asks[: self.params.loop.max_requests],
        )
