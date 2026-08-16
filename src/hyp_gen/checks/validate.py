"""Deterministic validation, run against the graph rather than against a model.

Two things get checked here, and neither needs an API key:

1. *Structural sanity* of the candidate itself -- does the shape still hold in
   this round's graph, and is it actually a hypothesis rather than a restated
   fact.
2. *Citation legality* of anything the model wrote -- every id it cites must
   exist in the evidence pack it was given. A model that cites `L7` when `L7`
   was never shown to it has stopped reporting and started remembering, and
   that is exactly the failure this system exists to prevent.

An ``error`` marks the hypothesis as blocked and keeps it out of the ranked
record; a ``warning`` is printed and kept. Errors are reserved for things that
make the output untrustworthy, not merely weak.
"""

from __future__ import annotations

from hyp_gen.generate.candidates import Candidate
from hyp_gen.generate.evidence import EvidencePack
from hyp_gen.graph import GraphIndex
from hyp_gen.hypothesis import Articulation, Critique, ValidationIssue


def check_structure(
    index: GraphIndex, candidate: Candidate, pack: EvidencePack
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for tid in (candidate.subject, candidate.object):
        if tid not in index.things:
            issues.append(
                ValidationIssue(
                    code="unknown_thing",
                    detail=f"{tid} is not a node in this graph",
                    severity="error",
                )
            )

    for lid in candidate.link_ids:
        if lid not in index.links:
            issues.append(
                ValidationIssue(
                    code="unknown_link",
                    detail=f"{lid} is not a link in this graph",
                    severity="error",
                )
            )

    # A path has to actually connect. A broken chain means the enumerator and
    # the graph disagree, which is a bug worth failing loudly on.
    #
    # Where the chain is expected to *start* is motif-dependent, and getting
    # this wrong is expensive rather than merely wrong: `broken_path` is an
    # error, so a motif checked against the wrong origin has every one of its
    # hypotheses blocked before articulation, in every run, invisibly.
    #
    # `analogical_transfer` is the case. Its path is the donor's bridge edge
    # (donor -> object), not the receiver's, because the whole proposal is that
    # the receiver *lacks* that edge — see the comment in
    # `candidates._analogical_transfers`. Walking it from `subject` asks the
    # receiver to already have the link the hypothesis exists to propose.
    if candidate.motif == "analogical_transfer":
        origins = set(candidate.analogues) or {candidate.subject}
    else:
        origins = {candidate.subject}

    if candidate.path and candidate.path[0].src not in origins:
        issues.append(
            ValidationIssue(
                code="broken_path",
                detail=(
                    f"{candidate.path[0].link_id} starts at {candidate.path[0].src}, "
                    f"expected one of {', '.join(sorted(origins))}"
                ),
                severity="error",
            )
        )
    else:
        node = candidate.path[0].src if candidate.path else candidate.subject
        for edge in candidate.path:
            if edge.src != node:
                issues.append(
                    ValidationIssue(
                        code="broken_path",
                        detail=f"{edge.link_id} starts at {edge.src}, expected {node}",
                        severity="error",
                    )
                )
                break
            node = edge.dst

    # Whatever the origin, the chain must arrive where the hypothesis says it
    # does. Without this, relaxing the origin above would let an analogical
    # bridge point anywhere at all.
    if candidate.path and candidate.path[-1].dst != candidate.object:
        issues.append(
            ValidationIssue(
                code="broken_path",
                detail=(
                    f"the path ends at {candidate.path[-1].dst}, but the hypothesis "
                    f"is about {candidate.object}"
                ),
                severity="error",
            )
        )

    if candidate.motif != "condition_split" and index.links_between(
        candidate.subject, candidate.object
    ):
        issues.append(
            ValidationIssue(
                code="already_stated",
                detail=(
                    f"the graph already links {index.name(candidate.subject)} and "
                    f"{index.name(candidate.object)} directly; this is a restatement, "
                    "not a hypothesis"
                ),
                severity="error",
            )
        )

    if not pack.findings:
        issues.append(
            ValidationIssue(
                code="no_findings",
                detail="no verbatim findings back any link on this path",
                severity="warning",
            )
        )

    if candidate.motif == "gap_closure" and not candidate.path:
        issues.append(
            ValidationIssue(
                code="unconnected_gap",
                detail=(
                    "no confident path connects the gap's endpoints, so this rests "
                    "entirely on the graph's own suggestion"
                ),
                severity="warning",
            )
        )

    if candidate.motif == "condition_split" and len(candidate.conditions) < 2:
        issues.append(
            ValidationIssue(
                code="conditions_unstated",
                detail=(
                    "the link disagrees but its findings state fewer than two distinct "
                    "conditions, so the condition variable is a guess"
                ),
                severity="warning",
            )
        )
    return issues


def check_citations(
    pack: EvidencePack,
    articulation: Articulation | None,
    critique: Critique | None,
) -> list[ValidationIssue]:
    legal = pack.legal_ids()
    issues: list[ValidationIssue] = []

    def audit(where: str, cites: list[str]) -> None:
        for cite in cites:
            if cite not in legal:
                issues.append(
                    ValidationIssue(
                        code="illegal_citation",
                        detail=f"{where} cites {cite}, which was not in its evidence pack",
                        severity="error",
                    )
                )

    if articulation is not None:
        for i, claim in enumerate(articulation.claims):
            audit(f"claim[{i}]", claim.cites)
            if not claim.cites and not claim.inferred:
                issues.append(
                    ValidationIssue(
                        code="uncited_claim",
                        detail=(
                            f"claim[{i}] cites nothing but is not marked inferred: "
                            f"{claim.text[:80]}"
                        ),
                        severity="warning",
                    )
                )
    if critique is not None:
        for finding in critique.per_claim:
            audit(f"critique[{finding.claim_index}]", finding.cites)
    return issues
