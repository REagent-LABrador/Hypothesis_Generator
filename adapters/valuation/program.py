"""Bundle → LABrador `ProgramInput`. The handoff to the valuation stage.

`managed/program-strategy-valuation/` (LABrador) is the downstream evaluator: it
takes a provenance-aware program brief and returns rNPV, protected years, payer
access, patient affordability and a decision grade. This module is the adapter
that turns a hypothesis into something it will accept.

The whole design is one rule: **emit what the graph states, mark everything else
unsupported, and never fill a gap with a plausible number.** LABrador is built to
punish invented precision — an unsupported critical input forces
`NOT_DECISION_GRADE` rather than a confident answer — so the correct adapter is
one that hands it honest holes and lets its gates fire. A `NOT_DECISION_GRADE`
result naming twelve missing inputs is the *successful* output of this module.

Three things the graph cannot know, and this module therefore refuses to guess:

*Prices, populations and access.* A literature graph is about mechanism. It
contains no eligible-patient count, no coverage fraction, no net price. Every
such field is emitted empty, and the evidence records this module writes are
namespaced (``mechanism:``, ``finding:``, ``paper:``) so that they *cannot*
collide with the field names LABrador's evidence gates look up. Literature
provenance rides along in the audit trail; it can never clear a payer gate. See
``LABRADOR_GATE_KEYS`` and the test that enforces it.

*The analyst frame.* Currency, geography, line of therapy, route, the valuation
and launch years, and above all the patent filing year are decisions, not
findings. They come from a `ProgramFrame` the caller supplies. There is no
default frame: `--emit-programs` without one is an error, because a guessed
filing year silently moves the single most consequential number LABrador
computes.

*Which motif's path means what.* An `analogical_transfer` candidate's path is the
**donor's** bridge edge — it is evidence about the donor, not about the molecule
being proposed — so mechanism nodes are read motif-aware rather than by walking
`path` blindly. Getting this wrong is silent, and it is the same trap that once
marked every analogical hypothesis `broken_path` (see `validate.py`).

One shape rule worth stating on its own. Two hypotheses about the same molecule
are two *labels on one asset*, not two programs. Emitting them separately would
give one molecule two patent clocks and two development budgets — exactly the
double count LABrador's "an expansion does not reset the clock" rule exists to
prevent — so they are grouped into `initial_indication` + `expansion_indications`
instead.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from adapters.common import Bundle
from hyp_gen.hypothesis import Hypothesis

# --------------------------------------------------------------------------
# The reserved namespace
# --------------------------------------------------------------------------

LABRADOR_GATE_KEYS: frozenset[str] = frozenset(
    {
        # pricing._CRITICAL_INPUT_FIELDS, plus the list-price gate
        "incremental_qalys",
        "willingness_to_pay_per_qaly",
        "comparator_total_cost",
        "new_non_drug_total_cost",
        "expected_treatment_years",
        "annual_comparator_drug_cost",
        "annual_non_drug_cost_offsets",
        "annual_payer_budget_limit",
        "annual_manufacturer_cost",
        "required_gross_margin_fraction",
        "candidate_list_price",
        # engine._cashflow_evidence_status → population
        "population_inputs",
        "eligible_patients",
        "prevalent_backlog_patients",
        "annual_incident_patients",
        "diagnosed_fraction",
        "clinically_eligible_fraction",
        # → health-system access
        "access_curve",
        "coverage_fraction",
        "prior_authorization_pass_fraction",
        "initiation_fraction",
        "provider_capacity_fraction",
        "adoption_by_year",
        "annual_patient_oop",
        "universal_or_public_coverage",
        "patient_cost_share_fraction",
        # → commercial
        "commercial_inputs",
        "annual_persistence_rate",
        "dose_intensity",
        "gross_to_net_rate",
        "cogs_per_full_dose_patient",
        # → patent
        "patent_inputs",
        "filing_year",
        "extension_years",
        "regulatory_exclusivity_end_year",
    }
)
"""Evidence keys LABrador consults when deciding whether a critical input is
supported.

This module must never write one of them. A paper showing that pirfenidone
inhibits TGF-β1 is not evidence for an eligible-patient count, and a graph-derived
record landing under ``eligible_patients`` would let mechanism literature clear a
payer gate — turning `NOT_DECISION_GRADE` into a false `DECISION_GRADE`. Every key
this module emits carries a ``prefix:`` so the collision is structurally
impossible; ``_assert_namespaced`` checks it anyway, because a silent version of
this bug is unrecoverable downstream.
"""

EVIDENCE_PREFIXES = ("hypothesis", "mechanism", "finding", "paper", "frame")


class NamespaceViolation(RuntimeError):
    """An emitted evidence key could be mistaken for a LABrador gate field."""


def _assert_namespaced(keys: Iterable[str]) -> None:
    for key in keys:
        prefix, _, rest = key.partition(":")
        if not rest or prefix not in EVIDENCE_PREFIXES:
            raise NamespaceViolation(
                f"evidence key {key!r} is not namespaced; expected one of "
                f"{'/'.join(EVIDENCE_PREFIXES)}:<id>"
            )
        if key in LABRADOR_GATE_KEYS:
            raise NamespaceViolation(f"evidence key {key!r} collides with a LABrador gate field")


# --------------------------------------------------------------------------
# Evidence translation
# --------------------------------------------------------------------------

_GRADE_LADDER = ("UNSUPPORTED", "VERY_LOW", "LOW", "MODERATE", "HIGH")

_STUDY_GRADE = {
    "meta_analysis": "HIGH",
    "clinical_trial": "HIGH",
    "human_cohort": "MODERATE",
    "animal": "LOW",
    "test_tube": "LOW",
    "computational": "VERY_LOW",
    "review": "LOW",
}
"""hyp_gen study types → LABrador `EvidenceGrade`.

The ceiling mirrors ``EvidenceParams.study_weights``, but the two scales answer
different questions and are deliberately not the same numbers: hyp_gen's weight
asks "how much should this move a support score", LABrador's grade asks "may this
clear a decision gate". Only HIGH and MODERATE clear one, which puts animal and
test-tube work below the line — correct, since a mouse result should never make a
payer-facing input decision-grade.
"""

_SECONDARY_STUDY_TYPES = {"meta_analysis", "review"}


def _demote(grade: str, steps: int) -> str:
    if steps <= 0:
        return grade
    index = _GRADE_LADDER.index(grade)
    return _GRADE_LADDER[max(0, index - steps)]


def _citation(paper: dict[str, Any]) -> str:
    parts = [
        str(paper.get("first_author") or "unknown author"),
        str(paper.get("year") or "n.d."),
    ]
    if paper.get("journal"):
        parts.append(str(paper["journal"]))
    if paper.get("title"):
        parts.append(str(paper["title"]))
    return ", ".join(parts)


def evidence_from_paper(
    paper_id: str,
    paper: dict[str, Any],
    *,
    hedged: bool = False,
    secondhand: bool = False,
    accessed_at: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """One LABrador `EvidenceMetadata` for one paper behind one finding.

    Demotions apply hyp_gen's own discounts to LABrador's grade ladder: a hedged
    sentence, a preprint, or a paper reporting somebody else's result each cost
    one rung. They compose, so a hedged secondhand preprint lands on UNSUPPORTED
    and cannot clear anything.

    ``source_date`` is left unset on purpose. The graph carries a publication
    *year*, and writing it as a January-1 `date` would manufacture ten months of
    precision that no source states. The year survives in the citation string,
    where it is legible as a year.
    """

    study = str(paper.get("study_type") or "computational")
    grade = _STUDY_GRADE.get(study, "VERY_LOW")
    penalties = sum((hedged, secondhand, bool(paper.get("is_preprint"))))
    grade = _demote(grade, penalties)

    doi = paper.get("doi")
    evidence_type = (
        "SECONDARY_RESEARCH"
        if (study in _SECONDARY_STUDY_TYPES or secondhand)
        else "PRIMARY_RESEARCH"
    )
    if grade == "UNSUPPORTED":
        evidence_type = "UNSUPPORTED"

    marks = [m for m, on in (("hedged", hedged), ("secondhand", secondhand)) if on]
    if paper.get("is_preprint"):
        marks.append("preprint")
    detail = f" ({', '.join(marks)})" if marks else ""

    return {
        "source_id": paper_id,
        "source_url": f"https://doi.org/{doi}" if doi else None,
        "citation": _citation(paper),
        "source_date": None,
        "accessed_at": accessed_at,
        "evidence_type": evidence_type,
        "grade": grade,
        "synthetic": False,
        "notes": (f"{study}{detail}. {note}").strip(),
    }


def unsupported(reason: str) -> dict[str, Any]:
    """An input the graph does not contain and this module will not invent."""

    return {
        "source_id": None,
        "source_url": None,
        "citation": None,
        "source_date": None,
        "accessed_at": None,
        "evidence_type": "UNSUPPORTED",
        "grade": "UNSUPPORTED",
        "synthetic": False,
        "notes": reason,
    }


def analyst_assumption(reason: str) -> dict[str, Any]:
    """A frame value: a stated analyst choice, never dressed as an observation.

    LABrador treats ``ASSUMPTION`` as visible but non-clearing, which is exactly
    the status a launch year picked by a human deserves.

    ``source_id`` is required rather than decorative: LABrador refuses any graded
    evidence without one, and it is right to — an assumption with nothing naming
    where it came from is indistinguishable from a number somebody typed. The
    identifier here says "a human wrote this in the frame", which is the truth.
    """

    return {
        "source_id": "analyst_frame",
        "source_url": None,
        "citation": None,
        "source_date": None,
        "accessed_at": None,
        "evidence_type": "ASSUMPTION",
        "grade": "LOW",
        "synthetic": False,
        "notes": reason,
    }


# --------------------------------------------------------------------------
# The analyst frame
# --------------------------------------------------------------------------

Route = Literal[
    "ORAL",
    "SUBCUTANEOUS_SELF",
    "SUBCUTANEOUS_CLINIC",
    "INTRAMUSCULAR",
    "INTRAVENOUS",
    "OTHER",
]
Modality = Literal["SMALL_MOLECULE", "PEPTIDE"]


class ProgramFrame(BaseModel):
    """What a human must decide before a hypothesis can be valued.

    Nothing here is derivable from a literature graph. The four year fields have
    no defaults because each one moves the answer materially and a plausible
    guess is indistinguishable from a sourced value once it is in the JSON —
    ``filing_year`` most of all, since LABrador's protected window is measured
    from filing and a year of error is a year of exclusivity.

    Everything else defaults to ``"UNSPECIFIED"`` rather than to something
    reasonable. A blank that reads as blank is safer than a plausible
    placeholder: LABrador's comparable matcher scores an ``UNSPECIFIED`` line of
    therapy as a non-match and tiers the comparable down, which is the behaviour
    you want when nobody has actually said what line this is.
    """

    model_config = ConfigDict(extra="forbid")

    base_year: int
    valuation_year: int
    launch_year: int
    filing_year: int

    currency: str = "USD"
    geography: str = "UNSPECIFIED"
    therapeutic_area: str = "UNSPECIFIED"
    target_population: str = "UNSPECIFIED"
    line_of_therapy: str = "UNSPECIFIED"
    route: Route = "OTHER"
    current_stage: str = "unspecified"

    modality: Modality | None = None
    """Required only when the graph does not state one. See ``_modality``."""

    target: str | None = None
    """Overrides the mechanism node picked off the path."""

    expansion_launch_year: int | None = None
    """Second label's launch. Defaults to ``launch_year``; LABrador rejects an
    expansion that launches before the initial indication."""

    notes: str = ""

    @classmethod
    def template(cls) -> dict[str, Any]:
        """A frame with the four decisions left as `null`, so it fails loudly.

        Mirrors `labrador example`: a starter file you edit, not a default you
        accidentally ship. Validating it raises until a human fills the years in.
        """

        return {
            "_README": (
                "Analyst frame for hyp_gen → LABrador. Every value here is a human "
                "decision, not a graph finding. The four nulls are required; a "
                "guessed filing_year silently moves LABrador's protected window."
            ),
            "base_year": None,
            "valuation_year": None,
            "launch_year": None,
            "filing_year": None,
            "currency": "USD",
            "geography": "UNSPECIFIED",
            "therapeutic_area": "UNSPECIFIED",
            "target_population": "UNSPECIFIED",
            "line_of_therapy": "UNSPECIFIED",
            "route": "OTHER",
            "current_stage": "unspecified",
            "modality": None,
            "target": None,
            "expansion_launch_year": None,
            "notes": "",
        }

    @classmethod
    def load(cls, payload: dict[str, Any]) -> ProgramFrame:
        return cls.model_validate({k: v for k, v in payload.items() if not k.startswith("_")})


# --------------------------------------------------------------------------
# Reading a hypothesis structurally
# --------------------------------------------------------------------------

_DISEASE_KIND = "disease"
_MECHANISM_KINDS = ("protein", "gene")


def _things(hypothesis: Hypothesis) -> dict[str, dict[str, Any]]:
    return dict(hypothesis.evidence.get("things") or {})


def _kind(hypothesis: Hypothesis, thing_id: str) -> str:
    return str(_things(hypothesis).get(thing_id, {}).get("kind") or "")


def mechanism_nodes(hypothesis: Hypothesis) -> tuple[str, ...]:
    """Interior nodes of the proposed mechanism, read motif-aware.

    A `transitive_chain` or `gap_closure` path runs subject → … → object, so its
    interior nodes are the mechanism. A `condition_split` is one link with no
    interior. An `analogical_transfer` path is the **donor's** bridge edge: it
    starts at a molecule that is not the subject, and reading a target off it
    would attribute the donor's mechanism to the receiver. That motif returns
    nothing, and the caveat is recorded on the emitted program.
    """

    if hypothesis.motif == "analogical_transfer":
        return ()
    interior: list[str] = []
    for step in hypothesis.path:
        node = str(step.get("to") or "")
        if node and node != hypothesis.object and node not in interior:
            interior.append(node)
    return tuple(interior)


def _modality(hypothesis: Hypothesis, frame: ProgramFrame) -> tuple[str | None, str]:
    """(modality, why). ``None`` means the hypothesis cannot be valued.

    Only ``small_molecule`` maps automatically. A graph's ``protein`` node is
    almost always a *target* rather than a peptide drug, and reading it as
    `PEPTIDE` would be an inference presented as a finding — the failure mode
    this repo has already been burned by. If the molecule is a peptide, a human
    says so in the frame.
    """

    kind = _kind(hypothesis, hypothesis.subject)
    if kind == "small_molecule":
        return "SMALL_MOLECULE", f"graph states {hypothesis.subject} is a small_molecule"
    if frame.modality:
        return frame.modality, f"frame states {frame.modality}; graph kind is {kind or 'unknown'}"
    return None, (
        f"subject {hypothesis.subject} has graph kind {kind or 'unknown'}, which LABrador's "
        "Modality does not cover; set `modality` in the frame to value it"
    )


def _target(hypothesis: Hypothesis, frame: ProgramFrame) -> tuple[str, dict[str, Any]]:
    """(target, evidence). Prefers a protein/gene the path actually crosses."""

    if frame.target:
        return frame.target, analyst_assumption("target supplied by the analyst frame")
    things = _things(hypothesis)
    nodes = mechanism_nodes(hypothesis)
    for node in nodes:
        if things.get(node, {}).get("kind") in _MECHANISM_KINDS:
            name = str(things[node].get("name") or node)
            links = ", ".join(str(s.get("link")) for s in hypothesis.path if s.get("link"))
            return name, {
                **unsupported(""),
                "source_id": node,
                "evidence_type": "SECONDARY_RESEARCH",
                "grade": "LOW",
                "citation": f"{hypothesis.motif} path over {links or 'no links'}",
                "notes": (
                    f"{name} is the first protein/gene node on the proposed path. It is the "
                    "mechanism the graph crosses, not a validated drug target."
                ),
            }
    if nodes:
        node = nodes[0]
        name = str(things.get(node, {}).get("name") or node)
        return name, unsupported(
            f"no protein or gene node on this path; used the first interior node ({name}), "
            f"kind {things.get(node, {}).get('kind') or 'unknown'}"
        )
    return "UNSPECIFIED", unsupported(
        "no mechanism node available for this motif; set `target` in the frame"
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-") or "unspecified"


def _accessed_at(record: Bundle) -> str | None:
    """The graph's generation timestamp is the literature access date."""

    stamp = record.generated_at
    return stamp[:10] if stamp and len(stamp) >= 10 else None


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------


class Skipped(BaseModel):
    """A hypothesis that cannot be expressed as a program, and why.

    Recorded rather than dropped. A record of eight that yields two programs has
    told you something about the graph, and a silent filter would hide it.
    """

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    reason: str
    detail: str


def eligible(hypothesis: Hypothesis, frame: ProgramFrame) -> tuple[bool, str, str]:
    """Can LABrador value this hypothesis at all? (ok, reason, detail)"""

    if hypothesis.blocked:
        codes = ", ".join(i.code for i in hypothesis.issues if i.severity == "error")
        return False, "blocked", f"validation errors: {codes}"
    if _kind(hypothesis, hypothesis.object) != _DISEASE_KIND:
        return False, "object_is_not_a_disease", (
            f"LABrador values a program against an indication; {hypothesis.object_name} has "
            f"graph kind {_kind(hypothesis, hypothesis.object) or 'unknown'}"
        )
    modality, why = _modality(hypothesis, frame)
    if modality is None:
        return False, "modality_not_in_graph", why
    return True, "", ""


def _indication(
    hypothesis: Hypothesis,
    record: Bundle,
    frame: ProgramFrame,
    *,
    launch_year: int,
    role: str,
    explicit_name: str | None = None,
) -> dict[str, Any]:
    """One `IndicationInput`: identity from the graph, everything else empty.

    Population and access are emitted as fully unknown structures rather than
    omitted. LABrador reads the absence as an error-severity warning naming the
    exact field, which is a far more useful artifact than a validation failure —
    the pipeline runs end to end and the output *is* the gap list.
    """

    accessed = _accessed_at(record)
    indication_name = explicit_name or hypothesis.object_name
    evidence_note = (
        f"Hypothesis {hypothesis.id} ({hypothesis.motif}) supplies mechanism evidence only; "
        "it does not propose or identify the economic indication. The population label "
        f"{indication_name!r} is copied verbatim from valuation_frame.target_population. "
        "A process node is never relabelled as a disease."
        if explicit_name is not None
        else (
            f"{role} indication proposed by hypothesis {hypothesis.id} "
            f"({hypothesis.motif}); scores {hypothesis.scores}. The graph names the disease; "
            "it says nothing about population, access or price."
        )
    )
    evidence: dict[str, Any] = {
        f"hypothesis:{hypothesis.id}": {
            **unsupported(""),
            "source_id": f"{record.graph_id}@round{record.round}",
            "evidence_type": "SECONDARY_RESEARCH",
            "grade": "LOW",
            "accessed_at": accessed,
            "citation": hypothesis.provenance,
            "notes": evidence_note,
        }
    }
    for link_id, link in (hypothesis.evidence.get("links") or {}).items():
        evidence[f"mechanism:{link_id}"] = {
            **unsupported(""),
            "source_id": link_id,
            "evidence_type": "SECONDARY_RESEARCH",
            "grade": "LOW",
            "accessed_at": accessed,
            "citation": (
                f"{link.get('from_name')} --{link.get('how')}--> {link.get('to_name')} "
                f"[{link.get('state')}]"
            ),
            "notes": (
                f"recomputed support {link.get('recomputed_support')}; "
                f"graph-stated confidence {link.get('stated_confidence')}"
            ),
        }
    papers = hypothesis.evidence.get("papers") or {}
    for finding_id, finding in (hypothesis.evidence.get("findings") or {}).items():
        paper_id = str(finding.get("paper") or "")
        paper = papers.get(paper_id)
        if not paper:
            continue
        evidence[f"finding:{finding_id}"] = evidence_from_paper(
            paper_id,
            paper,
            hedged=bool(finding.get("hedged")),
            secondhand=not finding.get("is_own_result", True),
            accessed_at=accessed,
            note=f'says={finding.get("says")}; where={finding.get("where") or "unstated"}',
        )
    _assert_namespaced(evidence)

    return {
        "indication_id": f"{_slug(indication_name)}-{_slug(role)}",
        "name": indication_name,
        "therapeutic_area": frame.therapeutic_area,
        "target_population": frame.target_population,
        "line_of_therapy": frame.line_of_therapy,
        "geography": frame.geography,
        "currency": frame.currency,
        "launch_year": launch_year,
        "severity": None,
        "biomarker": None,
        "route": frame.route,
        "comparator_ids": [],
        "population": {
            "eligible_patients": None,
            "prevalent_backlog_patients": None,
            "annual_incident_patients": None,
            "diagnosed_fraction": None,
            "clinically_eligible_fraction": None,
            "overlap_with_initial_fraction": None,
            "cannibalization_fraction": None,
            "evidence": {},
            "assumptions": {
                "source": "hyp_gen",
                "note": (
                    "A literature knowledge graph contains no epidemiology. Every population "
                    "field is unknown and must be supplied from a registry or survey before "
                    "this program can be decision-grade."
                ),
            },
        },
        "access": {
            "payer_type": "UNKNOWN",
            "universal_or_public_coverage": False,
            "coverage_fraction": None,
            "prior_authorization_pass_fraction": None,
            "initiation_fraction": None,
            "provider_capacity_fraction": None,
            "patient_cost_share_fraction": None,
            "annual_patient_oop": None,
            "adoption_by_year": {},
            "restrictions": [],
            "evidence": {},
            "assumptions": {
                "source": "hyp_gen",
                "note": (
                    "Coverage, prior authorization, initiation and capacity are payer facts. "
                    "The graph has none of them."
                ),
            },
        },
        "income_bands": [],
        "evidence": evidence,
        "assumptions": {
            "source": "hyp_gen",
            "hypothesis_id": hypothesis.id,
            "motif": hypothesis.motif,
            "role": role,
            "indication_identity_source": (
                "valuation_frame.target_population"
                if explicit_name is not None
                else "hypothesis.object"
            ),
            "hops": hypothesis.hops,
            "scores": hypothesis.scores,
            "rank_score": hypothesis.rank_score,
            "verification_verdict": (
                hypothesis.verification.verdict if hypothesis.verification else None
            ),
            "verification_halted_at": (
                hypothesis.verification.halted_at if hypothesis.verification else None
            ),
            "caveats": list(hypothesis.caveats),
            "graph_caveat": (
                "The path is the donor's bridge edge, not the subject's: it is evidence about "
                "the analogue, not about this molecule."
                if hypothesis.motif == "analogical_transfer"
                else ""
            ),
            "asks": [a.model_dump(mode="json") for a in hypothesis.asks],
            "analyst_todo": [
                "population.* — eligible, backlog, incidence, diagnosed and eligible fractions",
                "access.* — coverage, prior auth, initiation, capacity, adoption curve",
                "income_bands — required for a decision-grade affordability result",
                "therapeutic_area / target_population / line_of_therapy — frame placeholders",
            ],
        },
    }


def focused_program_input(
    hypothesis: Hypothesis,
    record: Bundle,
    frame: ProgramFrame,
) -> dict[str, Any]:
    """Build one ROI program when focus is a biomarker/process, not a disease.

    The economic identity comes only from the explicit analyst frame. The
    hypothesis contributes its stable id, mechanism, evidence, and open asks;
    it never turns its process/object node into an indication.
    """

    missing: list[str] = []
    if not frame.target or not frame.target.strip():
        missing.append("target")
    if frame.modality is None:
        missing.append("modality")
    if not frame.target_population.strip() or frame.target_population.casefold() == "unspecified":
        missing.append("target_population")
    if (
        not frame.therapeutic_area.strip()
        or frame.therapeutic_area.casefold() == "unspecified"
    ):
        missing.append("therapeutic_area")
    if hypothesis.blocked:
        codes = ", ".join(
            issue.code for issue in hypothesis.issues if issue.severity == "error"
        )
        raise ValueError(f"canonical focused hypothesis is blocked: {codes}")
    if missing:
        raise ValueError(
            "valuation_frame is incomplete for a focused ROI program: "
            + ", ".join(missing)
        )

    mechanism = None
    if hypothesis.articulation is not None:
        mechanism = (
            hypothesis.articulation.mechanism.strip()
            or hypothesis.articulation.statement.strip()
            or None
        )
    if mechanism is None:
        path = [
            (
                f"{step.get('from_name') or step.get('from')} "
                f"--{step.get('how') or 'related_to'}--> "
                f"{step.get('to_name') or step.get('to')}"
            )
            for step in hypothesis.path
        ]
        mechanism = "; ".join(path) or (
            f"{hypothesis.subject_name} --focused_hypothesis--> "
            f"{hypothesis.object_name}"
        )

    program = program_input([hypothesis], record, frame)
    program.update(
        {
            "program_id": f"{record.graph_id}-{_slug(hypothesis.id)}",
            "program_name": f"{frame.target} in {frame.target_population}",
            "molecule_identifier": f"frame-target:{frame.target}:{frame.modality}",
            "initial_indication": _indication(
                hypothesis,
                record,
                frame,
                launch_year=frame.launch_year,
                role="initial",
                explicit_name=frame.target_population,
            ),
        }
    )
    program["assumptions"].update(
        {
            "focused_hypothesis_id": hypothesis.id,
            "focused_hypothesis_mechanism": mechanism,
            "economic_indication_source": "valuation_frame.target_population",
            "program_name_source": "valuation_frame.target + valuation_frame.target_population",
            "program_name_limitation": (
                "The frame target is an analyst-supplied target/program placeholder, not a "
                "discovered or nominated molecule."
            ),
            "molecule_identifier_source": "valuation_frame.target + valuation_frame.modality",
            "molecule_identifier_limitation": (
                "Target/program placeholder only; the valuation frame names a target and modality, "
                "not a discovered or nominated molecule."
            ),
        }
    )
    return program


def program_input(
    hypotheses: list[Hypothesis],
    record: Bundle,
    frame: ProgramFrame,
) -> dict[str, Any]:
    """One LABrador `ProgramInput` from one molecule's hypotheses.

    ``hypotheses`` share a subject. The first is the initial indication and the
    second, if present, is the label expansion — one asset, one patent clock, two
    labels. LABrador models only the first expansion, so any others are the
    caller's to report (see ``emit``).
    """

    initial, *rest = hypotheses
    modality, modality_why = _modality(initial, frame)
    target, target_evidence = _target(initial, frame)
    accessed = _accessed_at(record)

    evidence: dict[str, Any] = {
        "hypothesis:target": target_evidence,
        "hypothesis:modality": {
            **unsupported(modality_why),
            "source_id": initial.subject,
            "accessed_at": accessed,
        },
        "frame:analyst_inputs": analyst_assumption(
            "base year, valuation year, launch year, filing year, currency, geography, route, "
            "line of therapy and stage are analyst decisions supplied in the program frame, "
            f"not graph findings. {frame.notes}".strip()
        ),
    }
    _assert_namespaced(evidence)

    expansion_launch = frame.expansion_launch_year or frame.launch_year
    expansions = [
        _indication(h, record, frame, launch_year=expansion_launch, role="expansion")
        for h in rest[:1]
    ]

    return {
        "program_id": f"{record.graph_id}-{_slug(initial.subject_name)}",
        "program_name": f"{initial.subject_name} in {initial.object_name}",
        "target": target,
        "modality": modality,
        "molecule_identifier": f"{initial.subject}:{initial.subject_name}",
        "route": frame.route,
        "base_year": frame.base_year,
        "valuation_year": frame.valuation_year,
        "currency": frame.currency,
        "initial_indication": _indication(
            initial, record, frame, launch_year=frame.launch_year, role="initial"
        ),
        "expansion_indications": expansions,
        "patent": {
            "filing_year": frame.filing_year,
            "base_term_years": 20,
            "extension_years": 0,
            "regulatory_exclusivity_end_year": None,
            "evidence": {},
            "assumptions": {
                "source": "analyst frame",
                "note": (
                    "Filing year is an analyst input; the graph has no patent record. "
                    "The 20-year term runs from filing and a label expansion does not "
                    "restart it. Confirm against Orange Book before relying on the "
                    "protected window."
                ),
            },
        },
        "development": {
            "current_stage": frame.current_stage,
            "stage_costs": {},
            "stage_durations_years": {},
            "stage_success_probabilities": {},
            "stage_order": [],
            "program_probability_of_approval": None,
            "evidence": {},
            "assumptions": {
                "source": "hyp_gen",
                "note": (
                    "No development path is emitted. The decisive experiment named by the "
                    "hypothesis is the next stage, but its cost, duration and success "
                    "probability are not in the graph."
                ),
                "decisive_experiment": (
                    initial.articulation.decisive_experiment if initial.articulation else None
                ),
                "falsifier": (initial.articulation.falsifier if initial.articulation else None),
            },
        },
        "evidence": evidence,
        "assumptions": {
            "source": "hyp_gen",
            "graph_id": record.graph_id,
            "graph_round": record.round,
            "graph_question": record.question,
            "graph_coverage": record.coverage,
            "absence_warning": (
                "Novelty in the source record is discounted by this graph's coverage. A missing "
                "link means this search did not surface it, never that nobody has shown it."
            ),
            "hypothesis_ids": [h.id for h in hypotheses[:2]],
            # How ambitious the run that proposed this was. An analyst reading a
            # brief deserves to know whether it came off a craziness-0.1 record or
            # a craziness-0.9 one; the numbers downstream look identical either
            # way and the appetite behind them does not.
            "stance": (record.params or {}).get("stance", {}),
            "analyst_todo": [
                "patent.filing_year — confirm against Orange Book",
                "development.stage_costs / durations / success probabilities",
                "comparables — the graph contains no price of any basis",
            ],
        },
    }


class Emission(BaseModel):
    """Everything one record produced for the valuation stage."""

    model_config = ConfigDict(extra="forbid")

    graph_id: str
    programs: list[dict[str, Any]] = Field(default_factory=list)
    skipped: list[Skipped] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def emit(record: Bundle, frame: ProgramFrame) -> Emission:
    """Group a record by molecule and emit one `ProgramInput` per molecule."""

    emission = Emission(graph_id=record.graph_id)
    by_subject: dict[str, list[Hypothesis]] = {}

    for hypothesis in record.hypotheses:
        ok, reason, detail = eligible(hypothesis, frame)
        if not ok:
            emission.skipped.append(
                Skipped(hypothesis_id=hypothesis.id, reason=reason, detail=detail)
            )
            continue
        by_subject.setdefault(hypothesis.subject, []).append(hypothesis)

    for subject, group in by_subject.items():
        group.sort(key=lambda h: -h.rank_score)
        emission.programs.append(program_input(group, record, frame))
        # No silent caps: LABrador models exactly one expansion, so say out loud
        # which labels were left on the floor rather than quietly emitting two.
        for dropped in group[2:]:
            emission.skipped.append(
                Skipped(
                    hypothesis_id=dropped.id,
                    reason="labrador_two_label_limit",
                    detail=(
                        f"{subject} already has an initial and an expansion indication; "
                        "LABrador's cash-flow model covers two labels on one patent clock."
                    ),
                )
            )

    if not emission.programs:
        emission.notes.append(
            "No hypothesis in this record is shaped like a program. LABrador values an "
            "intervention against a disease; run --profile repurposing or valuation."
        )
    emission.notes.append(
        "Every emitted program is NOT_DECISION_GRADE by construction: population, access, "
        "price and development inputs are absent because a literature graph does not contain "
        "them. Run `labrador analyze` to get the itemised gap list."
    )
    return emission
