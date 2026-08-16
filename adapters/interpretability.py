"""The shared LABrador interpretability contract, built from one document.

Every LABrador module ships a common ``interpretability`` block so one UI can
ask the same six questions of any result: what was concluded, why, on which
evidence and assumptions, how each number was derived, what uncertainty
remains, and what would change the conclusion. This module is hyp_gen's
adapter onto that contract, and it obeys the same rules as every adapter
(see ``adapters.common``): pure over the document, no model calls, no claim
the document does not carry.

Two rules are load-bearing here:

- **Map, never recompute.** Every value is copied from the authoritative
  document. Where the contract wants a derivation, the *inputs* to the
  recorded derivation are emitted next to the recorded result, so a reader
  can reconstruct the number — the builder itself never produces a second,
  potentially drifting computation of it.
- **No hidden constant.** The rank score depends on two values the resolved
  params may not name: the default structure weight applied when
  ``selection.rank_weights`` omits it, and the motif prior. Both are emitted
  as assumptions and as step inputs, so the displayed rank reconstructs
  exactly from the block.

Unknown stays ``null``, and every null of consequence carries a limitation.
Scores are heuristics on 0..1 and are labeled so; nothing here is a
probability, and the block must never call one that.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from adapters.common import Bundle
from hyp_gen.hypothesis import Hypothesis

SCHEMA_VERSION = "1.0.0"

# The default applied by hyp_gen.generate.scoring.normalise() when the run's
# rank_weights omit "structure". Emitted, never silently applied: reconstruction
# of rank_score is impossible without it.
DEFAULT_STRUCTURE_WEIGHT = 0.15

Status = Literal["SUPPORTED", "QUALIFIED", "INCONCLUSIVE", "FAILED", "NOT_APPLICABLE"]
Basis = Literal["OBSERVED", "INFERRED", "MODELED", "SYNTHETIC"]
Direction = Literal["positive", "negative", "neutral", "mixed", "unknown"]
Grade = Literal["HIGH", "MODERATE", "LOW", "UNSUPPORTED"]
Severity = Literal["INFO", "WARNING", "ERROR"]


class Headline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Short UI title.")
    result: str = Field(description="Stable machine-readable result.")
    plain_language: str = Field(description="One-sentence human explanation.")
    status: Status
    basis: list[Basis] = Field(description="Where the conclusion comes from.")


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable machine id, metric.<name>. Never positional.")
    label: str
    value: float | None = Field(description="Null when genuinely unknown, never 0.")
    unit: str
    display: str = Field(description="Short, plain-text rendering. No HTML.")
    meaning: str = Field(description="Why this number matters. Heuristics say so.")
    direction: Direction
    evidence_ids: list[str]
    assumption_ids: list[str]


class StepInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Where in the document or params this value lives.")
    value: Any
    unit: str | None


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Any
    unit: str | None


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    method: str
    formula: str | None
    inputs: list[StepInput]
    result: StepResult
    evidence_ids: list[str]
    assumption_ids: list[str]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="evidence.<finding id> — the document's own id.")
    claim: str
    source_type: str
    source_id: str | None
    source_url: str | None
    locator: str | None
    quote: str | None = Field(description="Verbatim from the document, or null.")
    grade: Grade
    synthetic: bool


class Assumption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    path: str
    value: Any
    unit: str | None
    basis: str
    synthetic: bool


class Interval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: str
    low: float | None
    central: float | None
    high: float | None
    unit: str
    confidence_level: float | None


class Uncertainty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    intervals: list[Interval]
    seed: int | None
    draws: int | None
    limitations: list[str]


class Limitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="STABLE_MACHINE_CODE.")
    severity: Severity
    message: str
    field_path: str | None


class Counterfactual(BaseModel):
    model_config = ConfigDict(extra="forbid")

    change: str
    result: str
    meaning: str


class Lineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str
    input_paths: list[str]
    transformation: str


class Interpretability(BaseModel):
    """The shared contract. Every field is required; an empty array is legal
    only when genuinely not applicable, and a limitation must say why."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    headline: Headline
    metrics: list[Metric]
    steps: list[Step]
    evidence: list[Evidence]
    assumptions: list[Assumption]
    uncertainty: Uncertainty
    limitations: list[Limitation]
    counterfactuals: list[Counterfactual]
    lineage: list[Lineage]
    extensions: dict[str, Any] = Field(
        description="Module-specific structured data. The shared UI must not require it."
    )


# -- the builder ------------------------------------------------------------

_HEURISTIC = "Heuristic score on 0..1, not a probability."

_METRIC_MEANINGS: dict[str, tuple[str, Direction, str]] = {
    "support": (
        f"How well the recorded findings back every link of the chain, recomputed "
        f"from study type, hedging and independent groups. {_HEURISTIC}",
        "positive",
        "Evidence support",
    ),
    "novelty": (
        f"Distance from what the graph already states, discounted by how entitled "
        f"this search is to claim absence. {_HEURISTIC}",
        "positive",
        "Novelty",
    ),
    "testability": (
        f"Whether the chain touches kinds an experiment can manipulate. {_HEURISTIC}",
        "positive",
        "Testability",
    ),
    "contradiction_risk": (
        f"Share of links with counter-findings or weak bases. {_HEURISTIC}",
        "negative",
        "Contradiction risk",
    ),
    "structure": (
        f"Degree-weighted path strength relative to the run's strongest candidate. "
        f"{_HEURISTIC}",
        "positive",
        "Structural strength",
    ),
    "absence_reliability": (
        f"How entitled this graph is to claim a link is absent, from coverage depth "
        f"and truncation. {_HEURISTIC}",
        "neutral",
        "Absence reliability",
    ),
}

_STRONG_STUDIES = {"meta_analysis", "clinical_trial", "human_cohort"}


def _grade(finding: dict, paper: dict | None, groups: int) -> Grade:
    """Deterministic display grade from fields the document records.

    Not a new judgement: hedged or secondhand findings grade LOW, firsthand
    results from strong human study types grade MODERATE, and HIGH is reserved
    for links replicated by at least two independent groups — the same
    distinctions the scoring already priced in.
    """
    if finding.get("hedged") or not finding.get("is_own_result"):
        return "LOW"
    paper = paper or {}
    if paper.get("study_type") not in _STRONG_STUDIES or paper.get("is_preprint"):
        return "LOW"
    return "HIGH" if groups >= 2 else "MODERATE"


def _says(claim: str, says: str | None) -> str:
    if says == "no":
        return f"Evidence against '{claim}'."
    if says == "no_effect":
        return f"No effect observed for '{claim}'."
    return claim


def _plain_language(h: Hypothesis) -> str:
    if h.articulation is not None:
        return h.articulation.statement
    return (
        f"Structural candidate ({h.motif}): the graph pattern proposes an "
        f"unstated relation between {h.subject_name} and {h.object_name}; "
        "no model has articulated it yet."
    )


def _status(h: Hypothesis) -> Status:
    if h.blocked:
        return "FAILED"
    verdict = h.verification.verdict if h.verification else None
    if verdict == "rejected":
        return "FAILED"
    if h.articulation is None:
        # A structural candidate is a shape, not a stated finding; it may not
        # present as SUPPORTED however well its links score.
        return "QUALIFIED" if verdict in ("verified", "qualified") else "INCONCLUSIVE"
    if verdict == "verified":
        return "SUPPORTED"
    if verdict == "qualified":
        return "QUALIFIED"
    return "INCONCLUSIVE"


def _run_mode(record: Bundle, h: Hypothesis) -> str:
    if record.counts.get("model_calls", 0) == 0 and h.articulation is None:
        return "DRY_RUN"
    ranking = record.params.get("ranking") or {}
    verification = record.params.get("verification") or {}
    if ranking.get("critique") and verification.get("enabled"):
        return "FULL"
    return "MODEL_ASSISTED"


def build(record: Bundle) -> Interpretability:
    """The interpretability block for a bundle's winning hypothesis.

    The winner is the highest ``rank_score`` in the bundle — for the usual
    single-document bundle, the one hypothesis the run selected. The rest of
    the bundle becomes the candidate ledger in ``extensions``.
    """
    winner = max(record.hypotheses, key=lambda h: h.rank_score)
    losers = [h for h in record.hypotheses if h is not winner]

    params = record.params
    selection = params.get("selection") or {}
    rank_weights = dict(selection.get("rank_weights") or {})
    motif_weights = (params.get("motifs") or {}).get("weights") or {}
    evidence_params = params.get("evidence") or {}
    support_weights = evidence_params.get("support_weights") or {}
    novelty_params = params.get("novelty") or {}
    profile = ((params.get("stance") or {}).get("profile")) or "default"

    limitations: list[Limitation] = []

    def limit(code: str, severity: Severity, message: str, field_path: str | None = None) -> None:
        limitations.append(
            Limitation(code=code, severity=severity, message=message, field_path=field_path)
        )

    # -- assumptions: every constant the ranking used, none left in code ----
    assumptions: list[Assumption] = []
    param_basis = f"Resolved run parameter (profile '{profile}'); configured, not measured."

    def assume(id_: str, path: str, value: Any, unit: str | None, basis: str = param_basis) -> str:
        assumptions.append(
            Assumption(id=id_, path=path, value=value, unit=unit, basis=basis, synthetic=False)
        )
        return id_

    rank_assumption_ids = [
        assume(
            f"assumption.rank_weight.{axis}",
            f"provenance.params.selection.rank_weights.{axis}",
            rank_weights.get(axis),
            "weight",
        )
        for axis in ("support", "novelty", "testability", "contradiction_risk")
    ]
    structure_weight = rank_weights.get("structure", DEFAULT_STRUCTURE_WEIGHT)
    rank_assumption_ids.append(
        assume(
            "assumption.rank_weight.structure",
            "provenance.params.selection.rank_weights.structure",
            structure_weight,
            "weight",
            basis=(
                param_basis
                if "structure" in rank_weights
                else "Default structure weight the scorer applies when "
                "selection.rank_weights omits it (generate/scoring.py)."
            ),
        )
    )
    motif_prior = motif_weights.get(winner.motif)
    rank_assumption_ids.append(
        assume(
            "assumption.motif_prior",
            f"provenance.params.motifs.weights.{winner.motif}",
            motif_prior,
            "weight",
            basis=f"Motif prior for {winner.motif}: multiplies the weighted axis sum. " + param_basis,
        )
    )

    support_assumption_ids = [
        assume(
            f"assumption.support_weight.{part}",
            f"provenance.params.evidence.support_weights.{part}",
            support_weights.get(part),
            "weight",
        )
        for part in ("evidence_quality", "agreement", "independence")
    ]
    support_assumption_ids.append(
        assume(
            "assumption.chain_aggregation",
            "provenance.params.evidence.chain_aggregation",
            evidence_params.get("chain_aggregation"),
            None,
        )
    )
    support_assumption_ids.append(
        assume(
            "assumption.single_group_cap",
            "provenance.params.evidence.single_group_cap",
            evidence_params.get("single_group_cap"),
            "score",
            basis="Cap on any link's support while it rests on one research group. " + param_basis,
        )
    )
    support_assumption_ids.append(
        assume(
            "assumption.min_independent_groups",
            "provenance.params.evidence.min_independent_groups",
            evidence_params.get("min_independent_groups"),
            "groups",
        )
    )

    novelty_assumption_ids = [
        assume(
            f"assumption.novelty.{key}",
            f"provenance.params.novelty.{key}",
            novelty_params.get(key),
            "weight",
        )
        for key in (
            "hop_novelty",
            "gap_novelty_bonus",
            "searched_gap_bonus",
            "gap_confidence_cap",
            "popularity_penalty",
        )
    ]
    testability_assumption_ids = [
        assume(
            "assumption.testable_kinds",
            "provenance.params.selection.testable_kinds",
            selection.get("testable_kinds"),
            None,
        )
    ]
    risk_assumption_ids = [
        assume(
            "assumption.contradiction_weight",
            "provenance.params.evidence.contradiction_weight",
            evidence_params.get("contradiction_weight"),
            "weight",
        )
    ]

    # -- evidence: every finding the winner's pack carries ------------------
    findings: dict[str, dict] = winner.evidence.get("findings") or {}
    papers: dict[str, dict] = winner.evidence.get("papers") or {}
    things: dict[str, dict] = winner.evidence.get("things") or {}
    per_link = winner.evidence.get("per_link_support") or []
    groups_by_link = {entry["link_id"]: entry.get("groups", 0) for entry in per_link}

    names = {tid: (t or {}).get("name") or tid for tid, t in things.items()}
    for hop in winner.path:
        names.setdefault(hop["from"], hop["from_name"])
        names.setdefault(hop["to"], hop["to_name"])

    def finding_link(fid: str) -> str | None:
        for link_id, link in (winner.evidence.get("links") or {}).items():
            sides = (link.get("yes") or []) + (link.get("no") or []) + (link.get("no_effect") or [])
            if fid in sides:
                return link_id
        return None

    evidence_items: list[Evidence] = []
    missing_source_id = False
    for fid, finding in findings.items():
        paper = papers.get(finding.get("paper") or "")
        doi = (paper or {}).get("doi")
        if doi is None:
            missing_source_id = True
        claim = _says(
            f"{names.get(finding.get('from'), finding.get('from'))} "
            f"{finding.get('how')} "
            f"{names.get(finding.get('to'), finding.get('to'))}",
            finding.get("says"),
        )
        link_id = finding_link(fid)
        evidence_items.append(
            Evidence(
                id=f"evidence.{fid}",
                claim=claim,
                source_type="publication",
                source_id=f"doi:{doi}" if doi else None,
                source_url=f"https://doi.org/{doi}" if doi else None,
                locator=None,
                quote=finding.get("quote"),
                grade=_grade(finding, paper, groups_by_link.get(link_id or "", 0)),
                synthetic=False,
            )
        )
    if missing_source_id:
        limit(
            "MISSING_SOURCE_IDENTIFIER",
            "WARNING",
            "At least one cited paper carries no DOI in the graph; its source_id "
            "and source_url are null rather than guessed.",
            "interpretability.evidence",
        )
    limit(
        "NO_SOURCE_LOCATORS",
        "INFO",
        "The knowledge graph records verbatim sentences but no page, figure or "
        "section locators, so every locator is null.",
        "interpretability.evidence",
    )
    limit(
        "EVIDENCE_GRADES_DERIVED",
        "INFO",
        "Grades are derived deterministically from the recorded study type, "
        "hedging, firsthand status and independent-group count — they are a "
        "display mapping, not a new appraisal.",
        "interpretability.evidence",
    )

    evidence_ids_all = [e.id for e in evidence_items]

    def evidence_for_link(link_id: str, sides: tuple[str, ...]) -> list[str]:
        link = (winner.evidence.get("links") or {}).get(link_id) or {}
        ids = [fid for side in sides for fid in (link.get(side) or [])]
        return [f"evidence.{fid}" for fid in ids if f"evidence.{fid}" in set(evidence_ids_all)]

    path_link_ids = [hop["link"] for hop in winner.path]
    support_evidence = [
        eid for lid in path_link_ids for eid in evidence_for_link(lid, ("yes", "no", "no_effect"))
    ]
    against_evidence = [
        eid for lid in path_link_ids for eid in evidence_for_link(lid, ("no", "no_effect"))
    ]

    # -- metrics ------------------------------------------------------------
    per_metric_refs: dict[str, tuple[list[str], list[str]]] = {
        "support": (support_evidence, support_assumption_ids),
        "novelty": ([], novelty_assumption_ids),
        "testability": ([], testability_assumption_ids),
        "contradiction_risk": (against_evidence, risk_assumption_ids),
        "structure": ([], []),
        "absence_reliability": ([], []),
    }
    metrics: list[Metric] = []
    for key, (meaning, direction, label) in _METRIC_MEANINGS.items():
        value = winner.scores.get(key)
        if value is None:
            limit(
                "NULL_SCORE",
                "WARNING",
                f"The document records no '{key}' score; the value is null, not 0.",
                f"hypothesis.scores.{key}",
            )
        ev_ids, as_ids = per_metric_refs[key]
        metrics.append(
            Metric(
                id=f"metric.{key}",
                label=label,
                value=value,
                unit="score",
                display=f"{value:.3f}" if value is not None else "not recorded",
                meaning=meaning,
                direction=direction,  # type: ignore[arg-type]
                evidence_ids=list(dict.fromkeys(ev_ids)),
                assumption_ids=as_ids,
            )
        )
    metrics.append(
        Metric(
            id="metric.rank_score",
            label="Rank score",
            value=winner.rank_score,
            unit="score",
            display=f"{winner.rank_score:.4f}",
            meaning=(
                "Orders candidates on a page; it does not grade the science. "
                "Weighted sum of the axis scores times the motif prior. "
                + _HEURISTIC
            ),
            direction="neutral",
            evidence_ids=[],
            assumption_ids=rank_assumption_ids,
        )
    )
    untagged = [m.id for m in metrics if not (m.evidence_ids or m.assumption_ids)]
    if untagged:
        limit(
            "UNTAGGED_VALUE",
            "INFO",
            f"{', '.join(untagged)} derive from graph topology and coverage recorded "
            "in provenance, not from citable evidence or a configured assumption.",
            "interpretability.metrics",
        )

    # -- steps: the recorded derivations, inputs beside results -------------
    steps: list[Step] = []
    for entry in per_link:
        link_id = entry["link_id"]
        inputs = [
            StepInput(
                path=f"per_link_support.{link_id}.{part}",
                value=entry.get(part),
                unit="score",
            )
            for part in ("evidence_quality", "agreement", "independence")
        ]
        inputs += [
            StepInput(
                path=f"provenance.params.evidence.support_weights.{part}",
                value=support_weights.get(part),
                unit="weight",
            )
            for part in ("evidence_quality", "agreement", "independence")
        ]
        inputs.append(
            StepInput(path=f"per_link_support.{link_id}.groups", value=entry.get("groups"), unit="groups")
        )
        inputs.append(
            StepInput(
                path="provenance.params.evidence.single_group_cap",
                value=evidence_params.get("single_group_cap"),
                unit="score",
            )
        )
        steps.append(
            Step(
                id=f"step.link_support.{link_id}",
                label=f"Recompute support for link {link_id}",
                method="weighted evidence recomputation",
                formula=(
                    "support = w_evidence_quality*evidence_quality + w_agreement*agreement "
                    "+ w_independence*independence; capped at single_group_cap while "
                    "groups < min_independent_groups. Components are recorded rounded "
                    "to 3 decimals."
                ),
                inputs=inputs,
                result=StepResult(value=entry.get("support"), unit="score"),
                evidence_ids=evidence_for_link(link_id, ("yes", "no", "no_effect")),
                assumption_ids=support_assumption_ids,
            )
        )
    if per_link:
        aggregation = evidence_params.get("chain_aggregation")
        steps.append(
            Step(
                id="step.chain_support",
                label="Aggregate per-link support over the chain",
                method=f"chain aggregation ({aggregation})",
                formula={
                    "weakest": "support = min(per-link support)",
                    "mean": "support = mean(per-link support)",
                    "noisy_or": "support = 1 - Π(1 - per-link support)",
                }.get(aggregation or ""),
                inputs=[
                    StepInput(
                        path=f"per_link_support.{entry['link_id']}.support",
                        value=entry.get("support"),
                        unit="score",
                    )
                    for entry in per_link
                ],
                result=StepResult(value=winner.scores.get("support"), unit="score"),
                evidence_ids=list(dict.fromkeys(support_evidence)),
                assumption_ids=["assumption.chain_aggregation"],
            )
        )
    rank_inputs = [
        StepInput(path=f"hypothesis.scores.{axis}", value=winner.scores.get(axis), unit="score")
        for axis in ("support", "novelty", "testability", "contradiction_risk", "structure")
    ]
    rank_inputs += [
        StepInput(
            path=f"provenance.params.selection.rank_weights.{axis}",
            value=rank_weights.get(axis) if axis != "structure" else structure_weight,
            unit="weight",
        )
        for axis in ("support", "novelty", "testability", "contradiction_risk", "structure")
    ]
    rank_inputs.append(
        StepInput(
            path=f"provenance.params.motifs.weights.{winner.motif}",
            value=motif_prior,
            unit="weight",
        )
    )
    steps.append(
        Step(
            id="step.rank_score",
            label="Combine the axes into the display rank",
            method="weighted sum with motif prior",
            formula=(
                "rank_score = round((w_support*support + w_novelty*novelty + "
                "w_testability*testability + w_contradiction_risk*contradiction_risk + "
                "w_structure*structure) * motif_prior, 4)"
            ),
            inputs=rank_inputs,
            result=StepResult(value=winner.rank_score, unit="score"),
            evidence_ids=[],
            assumption_ids=rank_assumption_ids,
        )
    )

    # -- headline -----------------------------------------------------------
    basis: list[Basis] = ["OBSERVED", "INFERRED"]
    if winner.articulation is not None:
        basis.append("MODELED")
    headline = Headline(
        title=f"{winner.subject_name} → {winner.object_name}",
        result=winner.id,
        plain_language=_plain_language(winner),
        status=_status(winner),
        basis=basis,
    )

    # -- limitations from the document's own warnings -----------------------
    if winner.articulation is None:
        limit(
            "STRUCTURAL_CANDIDATE_NOT_ARTICULATED",
            "WARNING",
            "Dry run: the candidate is a graph structure that was never articulated "
            "into a testable statement by a model; there is no falsifier or decisive "
            "experiment on record.",
            "hypothesis.articulation",
        )
    else:
        limit(
            "MODEL_IDENTITY_NOT_RECORDED",
            "INFO",
            "The document does not record which model or prompt revision produced "
            "the articulation; extensions.model carries nulls rather than a guess.",
            "interpretability.extensions.model",
        )
    for issue in winner.issues:
        limit(
            issue.code.upper(),
            "ERROR" if issue.severity == "error" else "WARNING",
            issue.detail,
            "hypothesis.issues",
        )
    if winner.verification:
        for gate in winner.verification.gates:
            if gate.status in ("warn", "fail") and gate.summary:
                seen = {i.detail for i in winner.issues}
                if gate.summary not in seen:
                    limit(
                        f"GATE_{gate.name.upper()}_{gate.status.upper()}",
                        "ERROR" if gate.status == "fail" else "WARNING",
                        f"{gate.name} gate: {gate.summary}",
                        "hypothesis.verification",
                    )
        if winner.verification.halted_at:
            limit(
                "VERIFICATION_HALTED",
                "ERROR",
                f"Verification stopped at {winner.verification.halted_at}; every gate "
                "below it did not run and none of them may be read as passed.",
                "hypothesis.verification.halted_at",
            )
    for caveat in winner.caveats:
        limit("DOCUMENT_CAVEAT", "WARNING", caveat, "hypothesis.caveats")
    coverage = record.coverage
    if coverage.get("truncated") or coverage.get("depth") == "quick":
        limit(
            "COVERAGE_TRUNCATED",
            "WARNING",
            f"The search read {coverage.get('read')} of {coverage.get('found')} "
            "results; absence of a link is weak evidence of absence, and novelty "
            "is discounted for it.",
            "provenance.coverage",
        )
    limit(
        "HEURISTIC_SCORES",
        "INFO",
        "support, novelty, testability, contradiction_risk, structure and "
        "rank_score are deterministic heuristics on 0..1, not probabilities.",
        "interpretability.metrics",
    )
    if not losers:
        limit(
            "CANDIDATE_LEDGER_LIMITED",
            "INFO",
            f"The document carries only the winner; {record.provenance.considered} "
            "candidate(s) were assembled and ranked before selection, but their "
            "scores are not in this bundle.",
            "interpretability.extensions.candidates",
        )

    # -- counterfactuals ----------------------------------------------------
    counterfactuals: list[Counterfactual] = []
    if winner.articulation is not None:
        counterfactuals.append(
            Counterfactual(
                change=f"Observe the recorded falsifier: {winner.articulation.falsifier}",
                result="The hypothesis is falsified.",
                meaning="This is the single observation the articulation names as fatal.",
            )
        )
        if winner.articulation.decisive_experiment:
            counterfactuals.append(
                Counterfactual(
                    change=f"Run the decisive experiment: {winner.articulation.decisive_experiment}",
                    result="The outcome discriminates between this hypothesis and its alternative.",
                    meaning="The cheapest recorded experiment that would settle the claim.",
                )
            )
    for ask in [*winner.asks, *record.asks]:
        counterfactuals.append(
            Counterfactual(
                change=f"Graph builder performs {ask.ask} on {ask.target}",
                result=(
                    "The evidence base changes and the scores would be recomputed; "
                    "the direction of the change is unknown until the search runs."
                ),
                meaning=ask.reason or "The recorded request that would most move this hypothesis.",
            )
        )
        break  # one ask is enough for the headline counterfactual; the rest stay in asks
    if not counterfactuals:
        limit(
            "NO_COUNTERFACTUALS_RECORDED",
            "INFO",
            "No falsifier, decisive experiment or graph-builder ask is on record "
            "for this run, so no counterfactual can be stated without inventing one.",
            "interpretability.counterfactuals",
        )

    # -- lineage ------------------------------------------------------------
    lineage = [
        Lineage(
            output_path="hypothesis.rank_score",
            input_paths=[
                "hypothesis.scores",
                "provenance.params.selection.rank_weights",
                f"provenance.params.motifs.weights.{winner.motif}",
            ],
            transformation="Weighted sum of the axis scores, times the motif prior (step.rank_score).",
        ),
        Lineage(
            output_path="hypothesis.scores.support",
            input_paths=[
                "hypothesis.evidence.findings",
                "hypothesis.evidence.papers",
                "provenance.params.evidence",
            ],
            transformation=(
                "Per-link support recomputed from findings and papers (study type, "
                "hedging, firsthand status, independent groups), then aggregated over "
                f"the chain by '{evidence_params.get('chain_aggregation')}'."
            ),
        ),
        Lineage(
            output_path="hypothesis.scores.novelty",
            input_paths=[
                "hypothesis.evidence.gap",
                "provenance.coverage",
                "provenance.params.novelty",
            ],
            transformation=(
                "Hop distance and gap bonuses, scaled by the graph's absence "
                "reliability and damped by node popularity. Node degrees live only "
                "in the graph, so this value cannot be re-derived from the document alone."
            ),
        ),
        Lineage(
            output_path="hypothesis.path",
            input_paths=["hypothesis.evidence.links"],
            transformation=(
                "Typed, degree-weighted traversal of the input graph; a reversed hop "
                "is marked and rendered against the stated arrow."
            ),
        ),
    ]
    limit(
        "NOVELTY_NOT_RECONSTRUCTIBLE",
        "INFO",
        "Novelty depends on node degrees recorded only in the input graph; the "
        "document carries the resulting score and its parameters, not those inputs.",
        "hypothesis.scores.novelty",
    )

    # -- extensions ---------------------------------------------------------
    extensions: dict[str, Any] = {
        "run_mode": _run_mode(record, winner),
        "motif": winner.motif,
        "graph_path": [dict(hop) for hop in winner.path],
        "verification": (
            winner.verification.model_dump(mode="json") if winner.verification else None
        ),
        "candidates": [
            {
                "id": h.id,
                "motif": h.motif,
                "rank_score": h.rank_score,
                "scores": dict(h.scores),
                "reason_not_selected": (
                    "blocked by an error-level validation issue"
                    if h.blocked
                    else f"ranked below the winner ({h.rank_score} < {winner.rank_score})"
                ),
            }
            for h in sorted(losers, key=lambda h: h.rank_score, reverse=True)
        ],
        "heuristic_scores": True,
    }
    if winner.articulation is not None:
        extensions["model"] = {"identifier": None, "prompt_revision": None}

    return Interpretability(
        headline=headline,
        metrics=metrics,
        steps=steps,
        evidence=evidence_items,
        assumptions=assumptions,
        uncertainty=Uncertainty(
            method="none",
            intervals=[],
            seed=None,
            draws=None,
            limitations=[
                "Scores are deterministic heuristics computed once from the document; "
                "no interval, distribution or confidence level exists for them, and "
                "none is invented here."
            ],
        ),
        limitations=limitations,
        counterfactuals=counterfactuals,
        lineage=lineage,
        extensions=extensions,
    )
