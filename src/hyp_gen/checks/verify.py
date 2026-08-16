"""Staged verification: one hypothesis walked through ordered gates.

``validate.py`` answers "is this check satisfied". This module answers "did this
hypothesis survive the process", which is a different question and the one a
reader actually has. It runs the checks in a fixed order, records what each one
found, and stops at the first failure that makes the rest meaningless.

Three properties are the point:

**Order is cost.** Every deterministic gate runs before the one gate that spends
model calls. A hypothesis whose citations are illegal, or whose entire evidence
base is one lab, is rejected for free rather than after three critic calls.

**A skip is not a pass.** When the process halts, the downstream gates are
recorded as skips naming the gate that stopped them. This is the failure mode
worth designing against: a report that shows five green checks because the sixth
never ran is worse than one that shows the halt, because it reads as more
verified rather than less.

**A gate reports, the params decide.** Which gates run, and which of them are
allowed to halt, come from ``params.verification``. A gate never decides its own
authority, so a run's strictness is a parameter you can diff between profiles
rather than a constant buried in a check.

Severity note: only ``structure`` and ``citations`` emit ``error`` issues, which
is what ``Hypothesis.blocked`` has always keyed on. The gates added here express
themselves through the gate table and the verification verdict instead, so
turning verification on does not silently start blocking hypotheses that earlier
runs published.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from hyp_gen.checks import validate
from hyp_gen.generate.candidates import Candidate
from hyp_gen.generate.evidence import EvidencePack
from hyp_gen.graph import GraphIndex
from hyp_gen.params import Params
from hyp_gen.hypothesis import (
    Articulation,
    Critique,
    GateResult,
    Hypothesis,
    ValidationIssue,
    Verification,
)

CriticRunner = Callable[[], list[Critique]]
"""Supplied by the caller so this module never touches a Judge, a budget, or a
retry. The pipeline hands in a closure that runs the lenses and absorbs
``BudgetExceeded``/``RefusalError``; a test hands in a lambda returning a fixed
list. Either way the gate only sees critiques."""


@dataclass
class GateContext:
    index: GraphIndex
    candidate: Candidate
    pack: EvidencePack
    params: Params
    articulation: Articulation | None = None
    critics: CriticRunner | None = None
    critiques: list[Critique] = field(default_factory=list)
    """Filled by the adversarial gate so the caller can read them back off the
    context rather than the gate having to mutate a Hypothesis."""

    verdict: str | None = None
    """Critic consensus, set by the adversarial gate. Distinct from the overall
    verification verdict, which is computed from every gate."""


# -- text helpers ----------------------------------------------------------
# Used only for the vacuous-falsifier check. Deliberately crude: this is a
# guard against "the hypothesis is false", not a semantic similarity model, and
# anything cleverer would start rejecting honest falsifiers.

_WORD = re.compile(r"[a-z0-9]+")

_VACUOUS = (
    re.compile(r"\bthe hypothesis (is|were) (false|wrong|incorrect|not true)\b"),
    re.compile(r"\bthis (is|were) (false|wrong|not true)\b"),
    re.compile(r"\b(if|when) (it|this|the hypothesis) (is|were) (false|wrong)\b"),
    re.compile(r"\bfail(s|ed)? to (hold|replicate)\b\s*$"),
    re.compile(r"\bno (relationship|association|effect|link) (is )?(found|observed)\b\s*$"),
)


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _contained_in(a: str, b: str) -> float:
    """Fraction of ``a``'s tokens that also appear in ``b``.

    Asymmetric, and the direction matters. The question is whether the
    *statement* has been swallowed whole by the falsifier -- "<statement>, but
    it is not true" contains every word of the statement, while measuring the
    other way dilutes that to nothing as soon as the falsifier adds a few words
    of its own. So callers ask "is the statement contained in the falsifier",
    never the reverse.
    """
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta) if ta else 0.0


def _cited_ids(articulation: Articulation) -> set[str]:
    return {cite for claim in articulation.claims for cite in claim.cites}


def _result(
    name: str, status: str, summary: str = "", issues: list[ValidationIssue] | None = None
) -> GateResult:
    return GateResult(name=name, status=status, summary=summary, issues=issues or [])


def _worst(
    issues: list[ValidationIssue],
    name: str,
    clean: str,
    fail_codes: frozenset[str] = frozenset(),
) -> GateResult:
    """Fold a list of validation issues into one gate result.

    ``fail_codes`` is what keeps gate status and issue severity independent, and
    that separation is load-bearing. ``severity`` decides whether
    ``Hypothesis.blocked`` is true, which controls whether a hypothesis reaches
    the record at all; gate status decides what the verification table says. A
    hypothesis resting on one lab should fail its gate and still be published
    with that failure visible -- if the new gates emitted ``error`` they would
    quietly delete hypotheses that every previous run reported.

    So only ``structure`` and ``citations`` produce errors (they come from
    ``validate.py`` and always did). Every gate added here names its fatal codes
    instead and leaves severity at ``warning``.
    """
    fatal = [i for i in issues if i.severity == "error" or i.code in fail_codes]
    if fatal:
        return _result(name, "fail", fatal[0].detail, issues)
    if issues:
        return _result(name, "warn", issues[0].detail, issues)
    return _result(name, "pass", clean)


# -- the gates -------------------------------------------------------------


def gate_structure(ctx: GateContext) -> GateResult:
    """Does the shape still hold in this round's graph, and is it a hypothesis
    at all rather than a restatement of something the graph already says."""
    issues = validate.check_structure(ctx.index, ctx.candidate, ctx.pack)
    hops = ctx.candidate.hops
    return _worst(issues, "structure", f"{hops} hop(s), path intact, not already stated")


def gate_citations(ctx: GateContext) -> GateResult:
    """Every id the model wrote must exist in the pack it was shown."""
    if ctx.articulation is None:
        return _result("citations", "skip", "not articulated")
    issues = validate.check_citations(ctx.pack, ctx.articulation, None)
    cited = _cited_ids(ctx.articulation)
    # "0 ids, all legal" is technically true and reads as reassurance. An
    # articulation that cites nothing has a grounding problem, which the next
    # gate is about -- this one should not imply it looked fine.
    clean = f"{len(cited)} ids, all legal" if cited else "nothing cited"
    return _worst(issues, "citations", clean)


def gate_consistency(ctx: GateContext) -> GateResult:
    """Is the articulation internally coherent, and anchored to the structure
    it came from.

    The failure this exists for is an articulation that is fluent, legally
    cited, and floating free of its own candidate -- every claim marked
    inferred, or every citation pointing at a node rather than at evidence. Both
    pass the citation gate, because nothing illegal was written.
    """
    art = ctx.articulation
    if art is None:
        return _result("consistency", "skip", "not articulated")
    if not art.claims:
        return _result(
            "consistency",
            "fail",
            "no claims: nothing here is separately checkable",
            [ValidationIssue(code="no_claims", detail="the articulation decomposed into nothing")],
        )

    fail_codes = frozenset({"ungrounded", "all_inferred"})

    issues: list[ValidationIssue] = []
    cited = _cited_ids(art)
    evidence_ids = cited & (set(ctx.pack.links) | set(ctx.pack.findings))

    if not evidence_ids:
        issues.append(
            ValidationIssue(
                code="ungrounded",
                detail=(
                    "no claim cites a link or a finding, so nothing here rests on "
                    "evidence — only on the existence of the nodes"
                ),
            )
        )
    elif not (cited & set(ctx.candidate.link_ids)):
        issues.append(
            ValidationIssue(
                code="off_path",
                detail=(
                    "no claim cites a link on this candidate's own path; the "
                    "articulation has drifted off the structure it was built from"
                ),
            )
        )

    if all(claim.inferred for claim in art.claims):
        issues.append(
            ValidationIssue(
                code="all_inferred",
                detail="every claim is marked inferred: this is reasoning, not a graph-backed hypothesis",
            )
        )

    seen: dict[str, int] = {}
    for i, claim in enumerate(art.claims):
        key = " ".join(sorted(_tokens(claim.text)))
        if key in seen:
            issues.append(
                ValidationIssue(
                    code="duplicate_claim",
                    detail=f"claim[{i}] restates claim[{seen[key]}], inflating the claim count",
                )
            )
        seen[key] = i

    # Direction laundering: the enumerator is allowed to walk a link backwards
    # (traversal.allow_edge_reversal), and the pack says so, but a claim that
    # cites that link without marking itself inferred is asserting the reverse
    # relation as something the graph states. It does not.
    reversed_links = {e.link_id for e in ctx.candidate.path if not e.forward}
    for i, claim in enumerate(art.claims):
        crossed = reversed_links & set(claim.cites)
        if crossed and not claim.inferred:
            issues.append(
                ValidationIssue(
                    code="reversed_link_asserted",
                    detail=(
                        f"claim[{i}] cites {', '.join(sorted(crossed))} in the reverse "
                        "of its stated direction but is not marked inferred"
                    ),
                )
            )

    if not art.novel_because.strip():
        issues.append(
            ValidationIssue(
                code="novelty_unstated",
                detail="novel_because is empty, so the claim to novelty is unexamined",
            )
        )

    return _worst(
        issues,
        "consistency",
        f"{len(art.claims)} claims, {len(evidence_ids)} grounded in evidence",
        fail_codes,
    )


def gate_independence(ctx: GateContext) -> GateResult:
    """Does the support come from more than one research group.

    Counted off the evidence pack rather than off the scores, so this checks the
    world the model was actually shown. One lab reporting a result five times is
    one result, and it is the single most common way a record looks better
    supported than it is.
    """
    params = ctx.params
    findings = ctx.pack.findings
    if not findings:
        return _result(
            "independence",
            "skip",
            "no findings on this path to attribute",
        )

    issues: list[ValidationIssue] = []
    own = {fid: f for fid, f in findings.items() if f["is_own_result"]}

    if not own and params.verification.require_primary_evidence:
        return _result(
            "independence",
            "fail",
            f"all {len(findings)} findings cite someone else's result — no primary evidence",
            [
                ValidationIssue(
                    code="no_primary_evidence",
                    detail=(
                        "every finding on this path is a secondhand citation, so the "
                        "graph read papers that read papers"
                    ),
                )
            ],
        )

    groups: dict[str, list[str]] = {}
    for fid, finding in own.items():
        paper = ctx.pack.papers.get(finding["paper"], {})
        author = paper.get("first_author")
        if author:
            groups.setdefault(author, []).append(fid)

    required = params.evidence.min_independent_groups
    count = len(groups)

    if count < required:
        shared = ", ".join(sorted(own)[:3])
        who = next(iter(groups), "unattributed")
        issues.append(
            ValidationIssue(
                code="insufficient_independence",
                detail=(
                    f"{shared} share first author {who} — {count} independent "
                    f"group(s), this run requires {required}"
                ),
            )
        )
    elif count == 1:
        who = next(iter(groups))
        issues.append(
            ValidationIssue(
                code="single_group",
                detail=f"all primary evidence here is from {who}; nothing replicates it",
            )
        )

    if own and all(f["hedged"] for f in own.values()):
        issues.append(
            ValidationIssue(
                code="all_hedged",
                detail="every primary finding is hedged, so no source states this outright",
            )
        )

    preprints = [
        fid
        for fid, f in own.items()
        if ctx.pack.papers.get(f["paper"], {}).get("is_preprint")
    ]
    if own and len(preprints) == len(own):
        issues.append(
            ValidationIssue(
                code="preprints_only",
                detail="every primary finding is from a preprint; none is peer reviewed",
            )
        )

    # On a clean pass the group count is the headline, because it is the number
    # a reader wants and it appears nowhere else in the report. On warn or fail
    # the issue detail wins, and every detail above states the count itself.
    return _worst(
        issues,
        "independence",
        f"{count} independent group(s), {len(own)} primary findings",
        frozenset({"insufficient_independence"}),
    )


def gate_falsifiability(ctx: GateContext) -> GateResult:
    """Can this be wrong, and is there a named thing that would show it.

    Runs before the adversarial gate on purpose: a hypothesis with no real
    falsifier is not something critics can usefully attack, and finding that out
    costs nothing here and three model calls there.
    """
    art = ctx.articulation
    if art is None:
        return _result("falsifiability", "skip", "not articulated")

    v = ctx.params.verification
    issues: list[ValidationIssue] = []
    falsifier = art.falsifier.strip()

    if not falsifier:
        issues.append(
            ValidationIssue(
                code="no_falsifier",
                detail="no falsifier: nothing stated here could turn out to be wrong",
            )
        )
    else:
        overlap = _contained_in(art.statement, falsifier)
        vacuous = any(p.search(falsifier.lower()) for p in _VACUOUS)
        if vacuous or overlap >= v.max_claim_overlap:
            issues.append(
                ValidationIssue(
                    code="vacuous_falsifier",
                    detail=(
                        f"the falsifier restates the hypothesis ({overlap:.0%} token "
                        "overlap) rather than naming an observation that would kill it"
                    ),
                )
            )
        elif len(falsifier) < v.min_falsifier_chars:
            issues.append(
                ValidationIssue(
                    code="terse_falsifier",
                    detail=f"the falsifier is {len(falsifier)} characters — too terse to act on",
                )
            )

    if not art.decisive_experiment.strip():
        issues.append(
            ValidationIssue(
                code="no_experiment",
                detail="no decisive experiment, so there is no cheapest way to settle this",
            )
        )
    if not art.predictions:
        issues.append(
            ValidationIssue(
                code="no_predictions",
                detail="no predictions stated, so only the falsifier can discriminate this",
            )
        )

    return _worst(
        issues,
        "falsifiability",
        "falsifier and experiment both concrete",
        frozenset({"no_falsifier", "vacuous_falsifier", "no_experiment"}),
    )


def gate_adversarial(ctx: GateContext) -> GateResult:
    """The one gate that costs money: critics, each with a different lens.

    It audits its own critiques' citations too. A critic that cites evidence it
    was never shown is exactly as untrustworthy as an articulator that does, and
    until now that check ran after the verdict had already been folded.
    """
    from hyp_gen.reasoning import reason  # local: keeps verify.py importable without the model stack

    if ctx.articulation is None:
        return _result("adversarial", "skip", "not articulated")
    if not ctx.params.ranking.critique:
        return _result("adversarial", "skip", "ranking.critique is off")
    if ctx.critics is None:
        return _result("adversarial", "skip", "no judge available")

    critiques = ctx.critics()
    if not critiques:
        return _result("adversarial", "skip", "no critiques were produced")

    ctx.critiques = critiques
    issues: list[ValidationIssue] = []
    for critique in critiques:
        issues.extend(validate.check_citations(ctx.pack, None, critique))

    verdict = reason.consensus(critiques, ctx.params)
    ctx.verdict = verdict
    lenses = ", ".join(c.lens or "general" for c in critiques)
    summary = f"{len(critiques)} lenses ({lenses}) → {verdict}"

    if any(i.severity == "error" for i in issues):
        return _result("adversarial", "fail", issues[0].detail, issues)
    if verdict in ("unsupported", "contradicted"):
        objection = next(
            (c.strongest_objection for c in critiques if c.verdict == verdict), ""
        )
        return _result("adversarial", "fail", f"{summary} — {objection}"[:200], issues)
    if verdict == "partly_supported" or issues:
        return _result("adversarial", "warn", summary, issues)
    return _result("adversarial", "pass", summary, issues)


GATES: dict[str, Callable[[GateContext], GateResult]] = {
    "structure": gate_structure,
    "citations": gate_citations,
    "consistency": gate_consistency,
    "independence": gate_independence,
    "falsifiability": gate_falsifiability,
    "adversarial": gate_adversarial,
}

_INTEGRITY = ("structure", "citations")
"""Halting here means the output cannot be trusted at all, which is a different
verdict from a hypothesis that was checked and found wanting."""


def _verdict(gates: list[GateResult], halted_at: str | None) -> str:
    if halted_at in _INTEGRITY:
        return "rejected"
    if halted_at or any(g.status == "fail" for g in gates):
        return "unverified"
    if any(g.status in ("warn", "skip") for g in gates):
        return "qualified"
    return "verified"


def verify(ctx: GateContext) -> Verification:
    """Walk the gates in order, stop at the first halting failure.

    Returns the record rather than mutating anything; ``apply`` is what writes
    the result onto a Hypothesis. Keeping those separate is what lets a test run
    the process over a hand-built context with no pipeline in sight.
    """
    v = ctx.params.verification
    results: list[GateResult] = []
    halted_at: str | None = None

    for name in v.gates:
        gate = GATES.get(name)
        if gate is None:
            # An unknown name is a params typo. Recording it as a skip beats
            # both crashing a finished run and silently checking less than the
            # params asked for.
            results.append(_result(name, "skip", "unknown gate name"))
            continue

        if halted_at is not None:
            results.append(_result(name, "skip", f"halted at {halted_at}"))
            continue

        result = gate(ctx)
        result.halting = name in v.halt_on
        results.append(result)
        if result.status == "fail" and result.halting:
            halted_at = name

    return Verification(
        verdict=_verdict(results, halted_at), gates=results, halted_at=halted_at
    )


def apply(hypothesis: Hypothesis, ctx: GateContext) -> Verification:
    """Run the process and write it onto the hypothesis.

    Issues are merged rather than appended: ``check_structure`` already ran at
    assembly, so re-running it here would otherwise double every structural
    warning in the report.
    """
    verification = verify(ctx)
    hypothesis.verification = verification

    seen = {(i.code, i.detail) for i in hypothesis.issues}
    for gate in verification.gates:
        for issue in gate.issues:
            if (issue.code, issue.detail) not in seen:
                seen.add((issue.code, issue.detail))
                hypothesis.issues.append(issue)

    if ctx.critiques:
        hypothesis.critiques = ctx.critiques
    if ctx.verdict is not None:
        hypothesis.verdict = ctx.verdict
    return verification
