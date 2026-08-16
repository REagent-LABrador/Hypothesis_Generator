"""OUTPUT CONTRACT. ``HypothesisDocument`` is what this app produces.

One run writes exactly one of these, as ``hypothesis.json``: a single
hypothesis -- the one that ranked first -- with the provenance needed to judge
it. Not a slate. ``RunResult`` in ``pipeline.py`` is the set-shaped working
state behind it, and it never reaches disk.

``schemas/SCHEMA.md`` is the authoritative contract -- annotated JSON, the
closed vocabularies, and a worked example from a real run. Consumers should read
that rather than this module; ``schemas/hypothesis.schema.json`` is the same
thing for a validator, generated from the models here by
``python tools/generate_schemas.py``.

Every claim carries its own citations, so a consumer (dataset support, ROI,
simulated preclinical) can attach to a single claim rather than to a whole
hypothesis. That granularity is the point: "the target is druggable" and "the
target matters in this disease" fail for different reasons and cost different
amounts to check.

The document is versioned by ``schema_version``. Adding an optional field is a
minor bump; removing a field, or narrowing what one may contain, is a major one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Verdict = Literal["supported", "partly_supported", "unsupported", "contradicted"]


class Claim(BaseModel):
    """One atomic, separately checkable assertion inside a hypothesis."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="A single assertion, stated so it can be true or false on its own.")
    cites: list[str] = Field(
        default_factory=list,
        description="Ids from the evidence pack only (link, finding, paper, thing, or gap ids).",
    )
    inferred: bool = Field(
        default=False,
        description="True when the graph does not state this and it is a step of reasoning.",
    )


class Articulation(BaseModel):
    """What the model is asked to produce from one structural candidate."""

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(description="The hypothesis in one testable sentence.")
    mechanism: str = Field(description="The proposed causal chain, in graph terms.")
    claims: list[Claim] = Field(description="The hypothesis decomposed into checkable pieces.")
    novel_because: str = Field(description="What the graph does NOT already state.")
    predictions: list[str] = Field(
        default_factory=list, description="Observations that should hold if this is true."
    )
    falsifier: str = Field(description="The single observation that would kill this.")
    decisive_experiment: str = Field(description="The cheapest experiment that discriminates.")
    assumptions: list[str] = Field(
        default_factory=list, description="What must be true but is not in the graph."
    )


class CritiqueFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_index: int
    verdict: Verdict
    reason: str
    cites: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    """The adversarial pass. Its job is to break the hypothesis, not polish it."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    strongest_objection: str
    unsupported_leaps: list[str] = Field(default_factory=list)
    per_claim: list[CritiqueFinding] = Field(default_factory=list)
    alternative_explanation: str = Field(
        default="", description="A duller reading of the same evidence, if one exists."
    )
    lens: str = Field(
        default="",
        description="Which angle this critic was told to attack from. Set by the harness, not the model.",
    )


class Comparison(BaseModel):
    """One pairwise debate in the tournament.

    Two hypotheses, one graph, one winner. Pairwise judgements are far more
    reliable than absolute scores -- a model asked "is this an 8 or a 9" is
    guessing, a model asked "which of these two is better supported" is not.
    """

    model_config = ConfigDict(extra="forbid")

    winner: Literal["A", "B"]
    margin: Literal["clear", "narrow"]
    reason: str
    decisive_evidence: list[str] = Field(
        default_factory=list, description="Ids that decided it. Pack ids only."
    )


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    detail: str
    severity: Literal["error", "warning"] = "warning"


GateStatus = Literal["pass", "warn", "fail", "skip"]
VerificationVerdict = Literal["verified", "qualified", "unverified", "rejected"]

_SUMMARY_WIDTH = 68


def _clip(text: str) -> str:
    return text if len(text) <= _SUMMARY_WIDTH else f"{text[: _SUMMARY_WIDTH - 1]}…"


class GateResult(BaseModel):
    """The outcome of one verification gate.

    ``summary`` is the one line that appears in the gate table, so it must say
    what happened rather than what was checked: "L4,L2 share first author — 1
    group" is a result, "checked independence" is not.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    status: GateStatus
    summary: str = ""
    issues: list[ValidationIssue] = Field(default_factory=list)
    halting: bool = Field(
        default=False,
        description="Whether a failure here stops the process. Set from params, not by the gate.",
    )


class Verification(BaseModel):
    """The full staged verification of one hypothesis.

    Distinct from ``Hypothesis.verdict``, which is only the adversarial
    critics' consensus. This is the whole process: what ran, what it found, and
    where it stopped.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: VerificationVerdict
    gates: list[GateResult] = Field(default_factory=list)
    halted_at: str | None = Field(
        default=None, description="Name of the gate that stopped the process, if one did."
    )

    def gate(self, name: str) -> GateResult | None:
        return next((g for g in self.gates if g.name == name), None)

    @property
    def failures(self) -> list[GateResult]:
        return [g for g in self.gates if g.status == "fail"]

    def table(self) -> str:
        """The gate table, fixed width, worst news legible at a glance.

        Summaries are clipped so the table stays scannable in a terminal and in
        a markdown code block. Nothing is lost by it: the full text of every
        finding is on ``GateResult.issues`` and is rendered in full by the
        report's validation section.
        """
        rows = [
            f"gate {i} {g.name:<16}{g.status.upper():<7}{_clip(g.summary)}".rstrip()
            for i, g in enumerate(self.gates, start=1)
        ]
        tail = f" (halted: {self.halted_at})" if self.halted_at else ""
        width = max((len(r) for r in rows), default=31)
        return "\n".join(
            [*rows, "─" * max(width, 31), f"VERDICT  {self.verdict}{tail}"]
        )


class Ask(BaseModel):
    """A request back to the graph builder. This is the loop closing."""

    model_config = ConfigDict(extra="forbid")

    graph_id: str
    ask: Literal["expand_node", "resolve_link", "test_gap", "new_question"]
    target: str
    depth: Literal["quick", "standard", "deep", "exhaustive"] = "standard"
    reason: str = ""
    for_hypothesis: str | None = None


class Hypothesis(BaseModel):
    """One fully assembled, inspectable hypothesis."""

    model_config = ConfigDict(extra="allow")

    id: str
    motif: str
    subject: str
    object: str
    subject_name: str
    object_name: str
    hops: int
    tags: list[str] = Field(default_factory=list)
    path: list[dict] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    rank_score: float = 0.0
    evidence: dict = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    articulation: Articulation | None = None
    critiques: list[Critique] = Field(default_factory=list)
    verdict: Verdict | None = Field(
        default=None, description="Consensus across critics, per refute_threshold."
    )
    verification: Verification | None = Field(
        default=None, description="The staged gate process. None means it never ran."
    )
    elo: float | None = Field(
        default=None, description="Set only when the tournament ran."
    )
    evolved_from: str | None = None
    evolution_operator: str | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    asks: list[Ask] = Field(default_factory=list)
    provenance: str = ""

    @property
    def blocked(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def critique(self) -> Critique | None:
        """The single harshest critique, for callers that want one."""
        order = {"contradicted": 0, "unsupported": 1, "partly_supported": 2, "supported": 3}
        return min(self.critiques, key=lambda c: order.get(c.verdict, 3), default=None)


SCHEMA_VERSION = "2.0"


class Provenance(BaseModel):
    """What the run was, so the hypothesis can be checked without the run.

    A hypothesis separated from what produced it is unfalsifiable in practice:
    support 0.5 from a cautious run and support 0.5 from an ambitious one are
    not the same claim about the world, and the score alone cannot tell them
    apart. So every document carries its own.
    """

    model_config = ConfigDict(extra="allow")

    graph_id: str = Field(description="Id of the input knowledge graph this run read.")
    round: int = Field(description="The graph's round number, copied from the input.")
    question: str = Field(description="The question the input graph was built to answer.")
    generated_at: str | None = Field(
        default=None, description="ISO 8601 timestamp of the run, if recorded."
    )
    params: dict = Field(
        default_factory=dict,
        description="The full resolved parameters. With the graph, these determine the output.",
    )
    coverage: dict = Field(
        default_factory=dict,
        description=(
            "What the input graph says about how thoroughly it searched. Read this "
            "before the scores: absence of evidence is not evidence of absence."
        ),
    )
    counts: dict[str, int] = Field(
        default_factory=dict,
        description="Run tallies: graph size, how many candidates were considered, model calls.",
    )
    considered: int = Field(
        default=0,
        description=(
            "How many hypotheses this run assembled and ranked before choosing the "
            "one in this document. Selection is real work; a reader should know it happened."
        ),
    )


class HypothesisDocument(BaseModel):
    """THE OUTPUT. One run over one knowledge graph produces exactly one of these.

    Written as ``hypothesis.json``. This is a single hypothesis -- the one that
    ranked first -- not a slate. A run still enumerates, scores and ranks many
    candidates internally, because that is how it knows which one is best, but
    only the winner crosses this boundary.

    The document is self-contained on purpose: ``provenance`` says what it was
    run on and with what stance, ``hypothesis`` is the claim with its evidence
    and its verification, and ``asks`` names the requests that would move it.
    Nothing here needs the graph to be re-read to be understood.

    Reading order for a consumer: ``schema_version``, then
    ``provenance.coverage``, then ``hypothesis.verification.verdict``, then the
    claim itself. A hypothesis carrying an error-level issue is not a finding.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Version of the hypothesis.json contract this file was written against.",
    )
    provenance: Provenance = Field(description="What produced this hypothesis.")
    hypothesis: Hypothesis = Field(description="The hypothesis itself: one claim, with its evidence.")
    asks: list[Ask] = Field(
        default_factory=list,
        description="Requests back to the graph builder that would move this hypothesis.",
    )
