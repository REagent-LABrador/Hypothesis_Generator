"""Bundle → web UI payload.

The second adapter in this package, next to ``valuation.py``: a pure function
of one ``Bundle`` producing the JSON a downstream consumer binds to. There the
consumer is LABrador; here it is a web UI rendering one card per hypothesis.
Being pure over the record means the payload can be produced — or reproduced —
from a saved ``hypotheses.json`` without re-running the pipeline, exactly like the
report modes.

Each card carries three things the UI displays directly:

- ``trace`` — the graph walk as one string,
  ``pirfenidone --inhibits--> myofibroblast differentiation --contributes_to--> …``,
  with a reversed hop rendered ``<--how--`` so a walk against the graph's
  stated arrow cannot pass for one along it.
- ``metrics`` — support, novelty, testability, and the rank used to order
  cards. Separate axes on purpose: a hypothesis with high support and low
  novelty is a known fact, and a UI that averages them will rank textbook
  statements first.
- ``highlights`` — short one-liners saying how the graph supports,
  contradicts, or qualifies the hypothesis. **Every one-liner is assembled
  from structured record fields and carries the ids it was assembled from in
  ``refs``**, so the UI can link each line to its evidence and a reader can
  check it. No highlight states anything the record does not; free prose from
  the model appears only clipped and only in the statement field.

Highlights are ordered weakest-first — failures, contradictions, cautions,
novelty, support — for the same reason the report is: a card that leads with
its support reads as advocacy.

The safety contract matches the report modes: a failure badge, a halted
verification, and an error-level validation issue always surface, both as
``status.flags`` and as highlights. On a truncated graph the payload-level
``warnings`` carry the absence-of-evidence notice; novelty one-liners must be
read next to it, which is why it is payload-level and not droppable per card.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Shared with the report on purpose: what counts as a failure, how a halt is
# worded, and how prose is clipped must not fork between views.
from adapters.common import Bundle, _clip, _failure_badges, _flags, _halt_note
from adapters.interpretability import Interpretability, build as build_interpretability
from hyp_gen.hypothesis import Hypothesis

SCHEMA_VERSION = "1.0"

# One-liners are for a card, not a paragraph. Clipping is sentence-aware and
# always marked (see report._clip), so a cut argument never looks finished.
_LINE = 160

Kind = Literal["failure", "contradiction", "caution", "novelty", "support"]

# Render order for kinds: weakest first.
_KIND_ORDER: dict[str, int] = {
    "failure": 0,
    "contradiction": 1,
    "caution": 2,
    "novelty": 3,
    "support": 4,
}


class Highlight(BaseModel):
    """One punchy line, plus the ids that make it checkable."""

    model_config = ConfigDict(extra="forbid")

    kind: Kind
    text: str
    refs: list[str] = Field(
        default_factory=list,
        description="Link/finding/gap ids this line was assembled from.",
    )


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    support: float | None = None
    novelty: float | None = None
    testability: float | None = None
    rank: float | None = Field(
        default=None,
        description="Orders cards on a page; it does not grade the science.",
    )


class Status(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verification: str | None = None
    critics: str | None = None
    flags: list[str] = Field(
        default_factory=list,
        description="Must-not-miss states: rejection badges, a halted gate run.",
    )


class Card(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    headline: str
    statement: str | None = Field(
        default=None,
        description="The articulated hypothesis, never clipped: it IS the claim.",
    )
    trace: str
    motif: str
    hops: int
    metrics: Metrics
    status: Status
    highlights: list[Highlight]


class WebPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    graph_id: str
    round: int
    question: str
    generated_at: str | None = None
    coverage: dict[str, Any]
    warnings: list[str] = Field(
        default_factory=list,
        description="Bundle-level facts every card must be read next to.",
    )
    hypotheses: list[Card]
    interpretability: Interpretability = Field(
        description=(
            "The shared LABrador interpretability contract for the bundle's "
            "winning hypothesis. Required: a payload without it does not "
            "validate. See adapters/interpretability.py and "
            "schemas/interpretability.schema.json."
        ),
    )


def trace(hypothesis: Hypothesis) -> str:
    """The walk as one string: ``a --how--> b <--how-- c``.

    A reversed hop keeps its own arrow direction. Flattening it to ``-->``
    would present a walk against the graph's stated edge as a walk along it —
    the trace exists precisely to make that distinction visible.
    """
    parts = [hypothesis.subject_name]
    for step in hypothesis.path:
        arrow = (
            f"<--{step['how']}--" if step["reversed"] else f"--{step['how']}-->"
        )
        parts.append(f"{arrow} {step['to_name']}")
    return " ".join(parts)


def _step_highlights(hypothesis: Hypothesis) -> list[Highlight]:
    """Support and contradiction lines, one per edge-with-evidence.

    Wording tracks the link's recorded state: ``agreed`` and ``single_source``
    say so explicitly, because "backed by f16" reads very differently once you
    know f16 is the only source there is.
    """
    links = hypothesis.evidence.get("links") or {}
    findings = hypothesis.evidence.get("findings") or {}
    out: list[Highlight] = []
    for step in hypothesis.path:
        link_id = step["link"]
        link = links.get(link_id) or {}
        claim = f"{step['from_name']} {step['how']} {step['to_name']}"
        yes = list(link.get("yes") or [])
        against = list(link.get("no") or []) + list(link.get("no_effect") or [])

        if yes:
            hedged = [f for f in yes if (findings.get(f) or {}).get("hedged")]
            ids = ", ".join(yes)
            if step["state"] == "single_source":
                text = f"One source only for '{claim}': {ids}."
            elif step["state"] == "agreed":
                text = f"The graph agrees: {claim} ({ids})."
            else:
                text = f"'{claim}' is backed by {ids} [{step['state']}]."
            if hedged:
                text += f" Hedged in the original: {', '.join(hedged)}."
            out.append(
                Highlight(
                    kind="support",
                    text=_clip(text, _LINE),
                    refs=[link_id, *yes],
                )
            )
        else:
            out.append(
                Highlight(
                    kind="caution",
                    text=_clip(
                        f"No verbatim finding backs '{claim}' in this pack "
                        f"({link_id}).",
                        _LINE,
                    ),
                    refs=[link_id],
                )
            )

        # A finding that argues the other way is exactly what a card must not
        # bury — one line per counter-finding, with its own sentence quoted.
        for finding_id in against:
            finding = findings.get(finding_id) or {}
            quote = finding.get("quote")
            text = f"{finding_id} pushes back on '{claim}'"
            text += f': "{quote}"' if quote else "."
            out.append(
                Highlight(
                    kind="contradiction",
                    text=_clip(text, _LINE),
                    refs=[link_id, finding_id],
                )
            )

        # The recomputation disagreeing with the graph's own confidence is a
        # fact about the evidence, not a style choice; surface it when material.
        stated = link.get("stated_confidence")
        recomputed = link.get("recomputed_support")
        if stated is not None and recomputed is not None:
            drift = recomputed - stated
            if abs(drift) >= 0.05:
                direction = "below" if drift < 0 else "above"
                out.append(
                    Highlight(
                        kind="caution",
                        text=(
                            f"{link_id} recomputes to {recomputed:.2f}, "
                            f"{abs(drift):.2f} {direction} the graph's stated "
                            f"{stated:.2f}."
                        ),
                        refs=[link_id],
                    )
                )
    return out


def _novelty_highlight(hypothesis: Hypothesis) -> Highlight:
    """Why this is new, said in one line and tied to the structure that says so.

    A gap note from the graph builder is preferred when present: it is the recorded sentence
    naming the absence. Otherwise the motif's own semantics are stated — each
    motif exists only where a particular thing is absent from the graph, so
    naming that absence invents nothing.
    """
    gap = hypothesis.evidence.get("gap") or {}
    if gap.get("id") and gap.get("note"):
        return Highlight(
            kind="novelty",
            text=_clip(f"Unstated in the graph ({gap['id']}): {gap['note']}.", _LINE),
            refs=[gap["id"]],
        )
    subject, obj = hypothesis.subject_name, hypothesis.object_name
    path_refs = [step["link"] for step in hypothesis.path]
    by_motif = {
        "transitive_chain": (
            f"No direct link joins {subject} to {obj} in this graph — "
            "the composed chain is the new claim."
        ),
        "gap_closure": (
            f"the graph builder flagged {subject} ↔ {obj} as implied by its links "
            "but stated by nobody."
        ),
        "analogical_transfer": (
            f"{subject} lacks the edge its analogue has — the transfer "
            "itself is the proposal."
        ),
        "condition_split": (
            "The graph holds both results; the split proposes the condition "
            "as the variable."
        ),
    }
    text = by_motif.get(
        hypothesis.motif, f"The graph does not state {subject} → {obj}."
    )
    return Highlight(kind="novelty", text=text, refs=path_refs)


def _card(hypothesis: Hypothesis, shared_caveats: frozenset[str]) -> Card:
    highlights: list[Highlight] = []

    # Failures first, and never softened: same badges the report shows.
    for badge in _failure_badges(hypothesis):
        highlights.append(
            Highlight(kind="failure", text=badge.replace("**", ""), refs=[])
        )
    for issue in hypothesis.issues:
        if issue.severity == "error":
            highlights.append(
                Highlight(
                    kind="failure",
                    text=_clip(f"{issue.code}: {issue.detail}", _LINE),
                    refs=[],
                )
            )

    halt = _halt_note(hypothesis)
    if halt:
        highlights.append(
            Highlight(kind="caution", text=halt.replace("**", ""), refs=[])
        )
    if hypothesis.verification:
        for gate in hypothesis.verification.gates:
            if gate.status == "warn" and gate.summary:
                highlights.append(
                    Highlight(
                        kind="caution",
                        text=_clip(f"{gate.name} gate: {gate.summary}", _LINE),
                        refs=[],
                    )
                )

    # Scoring notes are already one-liners written for exactly this purpose.
    for note in hypothesis.evidence.get("scoring_notes") or []:
        highlights.append(Highlight(kind="caution", text=_clip(note, _LINE), refs=[]))

    # Caveats every hypothesis shares are payload-level warnings; only what is
    # particular to this one lands on its card.
    for caveat in hypothesis.caveats:
        if caveat not in shared_caveats:
            highlights.append(
                Highlight(kind="caution", text=_clip(caveat, _LINE), refs=[])
            )

    highlights.append(_novelty_highlight(hypothesis))
    highlights.extend(_step_highlights(hypothesis))
    highlights.sort(key=lambda h: _KIND_ORDER[h.kind])

    return Card(
        id=hypothesis.id,
        headline=f"{hypothesis.subject_name} → {hypothesis.object_name}",
        statement=hypothesis.articulation.statement
        if hypothesis.articulation
        else None,
        trace=trace(hypothesis),
        motif=hypothesis.motif,
        hops=hypothesis.hops,
        metrics=Metrics(
            support=hypothesis.scores.get("support"),
            novelty=hypothesis.scores.get("novelty"),
            testability=hypothesis.scores.get("testability"),
            rank=hypothesis.rank_score,
        ),
        status=Status(
            verification=hypothesis.verification.verdict
            if hypothesis.verification
            else None,
            critics=hypothesis.verdict,
            flags=_flags(hypothesis),
        ),
        highlights=highlights,
    )


def emit(record: Bundle) -> WebPayload:
    """One payload per record. Pure: same record in, same payload out."""
    warnings: list[str] = []
    cov = record.coverage
    if cov.get("truncated") or cov.get("depth") == "quick":
        warnings.append(
            "Absence of a link is not evidence of absence: this search read "
            f"{cov.get('read')} of {cov.get('found')} results. Novelty is "
            "discounted for that."
        )

    shared: frozenset[str] = frozenset()
    if len(record.hypotheses) > 1:
        shared = frozenset.intersection(
            *(frozenset(h.caveats) for h in record.hypotheses)
        )
    if shared:
        # Keep the first hypothesis's ordering so the warning list is stable.
        warnings.extend(
            _clip(c, _LINE)
            for c in record.hypotheses[0].caveats
            if c in shared
        )

    return WebPayload(
        graph_id=record.graph_id,
        round=record.round,
        question=record.question,
        generated_at=record.generated_at,
        coverage=dict(cov),
        warnings=warnings,
        hypotheses=[_card(h, shared) for h in record.hypotheses],
        interpretability=build_interpretability(record),
    )
