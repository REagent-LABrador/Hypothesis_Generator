"""The handoff to LABrador.

Two layers. The structural tests below always run and cover the properties this
module is responsible for: namespacing, grade translation, motif-aware mechanism
reading, and the refusal to invent. The last test in the file validates a real
emission against LABrador's own Pydantic contracts, and skips when the package is
not installed -- a second, hand-maintained copy of that contract here would be an
untested one, which is the failure this repo already warns about.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyp_gen.graph import KnowledgeGraph
from hyp_gen.params import Params
from adapters.common import Bundle
from hyp_gen.hypothesis import Provenance
from conftest import bundle
from hyp_gen.pipeline import Generator
from adapters.valuation.program import (
    LABRADOR_GATE_KEYS,
    NamespaceViolation,
    ProgramFrame,
    _assert_namespaced,
    analyst_assumption,
    emit,
    evidence_from_paper,
    mechanism_nodes,
    program_input,
    unsupported,
)

FRAME = Path(__file__).resolve().parents[2] / "examples" / "analyst-frame.json"


@pytest.fixture
def frame() -> ProgramFrame:
    return ProgramFrame.load(json.loads(FRAME.read_text()))


@pytest.fixture
def record(graph: KnowledgeGraph) -> RunResult:
    params = Params.profile("repurposing")
    return bundle(Generator(graph=graph, params=params).run())


# -- the invariant ---------------------------------------------------------


def test_no_emitted_evidence_key_can_clear_a_labrador_gate(record: RunResult, frame: ProgramFrame):
    """The one bug that would be unrecoverable downstream.

    A paper about pirfenidone is not evidence for an eligible-patient count. If a
    graph-derived record landed under ``eligible_patients``, LABrador would read
    mechanism literature as a cleared payer gate and a NOT_DECISION_GRADE program
    would come back DECISION_GRADE -- silently, and with a real citation attached.
    """
    emission = emit(record, frame)
    assert emission.programs

    for program in emission.programs:
        keys = set(program["evidence"])
        for indication in [program["initial_indication"], *program["expansion_indications"]]:
            keys |= set(indication["evidence"])
            keys |= set(indication["population"]["evidence"])
            keys |= set(indication["access"]["evidence"])
        keys |= set(program["patent"]["evidence"]) | set(program["development"]["evidence"])

        assert keys, "a program with no evidence records has lost its provenance"
        assert not (keys & LABRADOR_GATE_KEYS)
        assert all(":" in key for key in keys)


def test_namespace_guard_rejects_a_bare_gate_key():
    with pytest.raises(NamespaceViolation):
        _assert_namespaced(["eligible_patients"])
    with pytest.raises(NamespaceViolation):
        _assert_namespaced(["bogus:x"])
    _assert_namespaced(["finding:f1", "mechanism:L2"])


# -- evidence translation --------------------------------------------------


def test_study_type_sets_the_ceiling_and_demotions_only_go_down():
    trial = evidence_from_paper("p1", {"study_type": "clinical_trial", "doi": "10.1/x"})
    assert trial["grade"] == "HIGH"
    assert trial["source_url"] == "https://doi.org/10.1/x"

    hedged = evidence_from_paper("p1", {"study_type": "clinical_trial"}, hedged=True)
    assert hedged["grade"] == "MODERATE"

    # The three demotions compose: a hedged, secondhand preprint of a trial
    # falls three rungs, from HIGH to VERY_LOW.
    buried = evidence_from_paper(
        "p1", {"study_type": "clinical_trial", "is_preprint": True}, hedged=True, secondhand=True
    )
    assert buried["grade"] == "VERY_LOW"

    # And the ladder has a floor rather than wrapping round it.
    floor = evidence_from_paper(
        "p1", {"study_type": "computational", "is_preprint": True}, hedged=True, secondhand=True
    )
    assert floor["grade"] == "UNSUPPORTED"
    assert floor["evidence_type"] == "UNSUPPORTED"


def test_bench_work_cannot_clear_a_decision_gate():
    """LABrador clears only HIGH/MODERATE. A mouse result must sit below that."""
    for study in ("animal", "test_tube", "computational", "review"):
        assert evidence_from_paper("p1", {"study_type": study})["grade"] not in {
            "HIGH",
            "MODERATE",
        }


def test_secondhand_findings_are_typed_as_secondary_research():
    record = evidence_from_paper("p1", {"study_type": "human_cohort"}, secondhand=True)
    assert record["evidence_type"] == "SECONDARY_RESEARCH"


def test_publication_year_is_never_written_as_a_date():
    """A year is not a January 1st. Inventing ten months of precision is the
    exact habit both halves of this pipeline exist to prevent."""
    record = evidence_from_paper("p1", {"study_type": "human_cohort", "year": 2019})
    assert record["source_date"] is None
    assert "2019" in record["citation"]


def test_graded_records_always_carry_a_source_identifier():
    """LABrador rejects graded evidence with nothing naming where it came from."""
    for record in (analyst_assumption("frame"), evidence_from_paper("p1", {})):
        if record["grade"] not in {"UNSUPPORTED", "SYNTHETIC"}:
            assert record["source_id"] or record["source_url"] or record["citation"]
    assert unsupported("nothing")["grade"] == "UNSUPPORTED"


def test_nothing_emitted_is_ever_marked_synthetic(record: RunResult, frame: ProgramFrame):
    """`synthetic` is LABrador's flag for fabricated demo data. Graph-derived
    provenance is weak, sometimes unsupported, but it is not fabricated, and
    mislabelling it would make a real citation look like a fixture."""
    blob = json.dumps([p for p in emit(record, frame).programs])
    assert '"synthetic": true' not in blob


# -- motif-aware structure -------------------------------------------------


def test_analogical_transfer_yields_no_mechanism_nodes(record: RunResult):
    """Its path is the donor's bridge edge. Reading a target off it would
    attribute the analogue's mechanism to the molecule being proposed."""
    analogical = [h for h in record.hypotheses if h.motif == "analogical_transfer"]
    assert analogical, "the repurposing profile should surface at least one"
    for hypothesis in analogical:
        assert mechanism_nodes(hypothesis) == ()


def test_chain_mechanism_nodes_exclude_the_endpoint(record: RunResult):
    chains = [h for h in record.hypotheses if h.motif == "transitive_chain"]
    assert chains
    for hypothesis in chains:
        nodes = mechanism_nodes(hypothesis)
        assert nodes
        assert hypothesis.object not in nodes


def test_donor_caveat_is_carried_onto_the_program(record: RunResult, frame: ProgramFrame):
    for program in emit(record, frame).programs:
        for indication in [program["initial_indication"], *program["expansion_indications"]]:
            if indication["assumptions"]["motif"] == "analogical_transfer":
                assert "donor" in indication["assumptions"]["graph_caveat"]


# -- the refusal to invent -------------------------------------------------


def test_population_and_access_are_emitted_empty(record: RunResult, frame: ProgramFrame):
    """A literature graph has no epidemiology and no payer facts. Emitting the
    structures unknown -- rather than omitting them -- is what lets LABrador run
    to completion and return the gap list instead of a validation failure."""
    for program in emit(record, frame).programs:
        for indication in [program["initial_indication"], *program["expansion_indications"]]:
            assert all(v is None for v in indication["population"].values() if not isinstance(v, dict))
            assert indication["access"]["coverage_fraction"] is None
            assert indication["access"]["adoption_by_year"] == {}
            assert indication["income_bands"] == []


def test_no_development_path_or_price_is_invented(record: RunResult, frame: ProgramFrame):
    for program in emit(record, frame).programs:
        assert program["development"]["stage_costs"] == {}
        assert program["development"]["program_probability_of_approval"] is None
        assert program["patent"]["base_term_years"] == 20
        assert program["patent"]["extension_years"] == 0


def test_frame_template_will_not_validate_until_a_human_fills_it_in():
    with pytest.raises(Exception):
        ProgramFrame.load(ProgramFrame.template())


def test_protein_subject_needs_an_explicit_modality(record: RunResult, frame: ProgramFrame):
    """A `protein` node in a literature graph is nearly always the target, not a
    peptide drug. Reading it as PEPTIDE would be an inference dressed as a
    finding, so the frame has to say so."""
    hypothesis = next(h for h in record.hypotheses).model_copy(deep=True)
    hypothesis.evidence["things"][hypothesis.subject]["kind"] = "protein"
    stub = Bundle(
        provenance=Provenance(graph_id="g", round=1, question="q"), hypotheses=[hypothesis])

    assert emit(stub, frame).skipped[0].reason == "modality_not_in_graph"
    assert emit(stub, frame.model_copy(update={"modality": "PEPTIDE"})).programs


# -- shape rules -----------------------------------------------------------


def test_one_molecule_becomes_one_program_with_two_labels(record: RunResult, frame: ProgramFrame):
    """Two hypotheses about one molecule are two labels on one asset. Emitting
    them as separate programs would give the molecule two patent clocks and two
    development budgets -- the double count LABrador's shared-clock rule exists
    to prevent."""
    emission = emit(record, frame)
    subjects = [p["molecule_identifier"] for p in emission.programs]
    assert len(subjects) == len(set(subjects))

    multi = [p for p in emission.programs if p["expansion_indications"]]
    assert multi, "pirfenidone appears twice in the repurposing record"
    program = multi[0]
    assert program["expansion_indications"][0]["launch_year"] >= program["initial_indication"][
        "launch_year"
    ]


def test_extra_labels_are_reported_not_dropped(frame: ProgramFrame, record: RunResult):
    """No silent caps: LABrador models one expansion, so a third label has to be
    said out loud rather than quietly discarded."""
    subject_hypotheses = [h for h in record.hypotheses if h.subject == "t1"]
    assert subject_hypotheses
    padded = [h.model_copy(deep=True, update={"id": f"{h.id}-copy{n}"}) for n in range(3) for h in subject_hypotheses[:1]]
    stub = Bundle(
        provenance=Provenance(graph_id="g", round=1, question="q"), hypotheses=[*subject_hypotheses, *padded]
    )
    reasons = {s.reason for s in emit(stub, frame).skipped}
    assert "labrador_two_label_limit" in reasons


def test_non_disease_endpoints_are_skipped_with_a_reason(record: RunResult, frame: ProgramFrame):
    hypothesis = next(h for h in record.hypotheses).model_copy(deep=True)
    hypothesis.evidence["things"][hypothesis.object]["kind"] = "process"
    stub = Bundle(
        provenance=Provenance(graph_id="g", round=1, question="q"), hypotheses=[hypothesis])

    skipped = emit(stub, frame).skipped
    assert skipped[0].reason == "object_is_not_a_disease"
    assert not emit(stub, frame).programs


def test_empty_slate_says_so_rather_than_returning_nothing(frame: ProgramFrame):
    emission = emit(Bundle(
        provenance=Provenance(graph_id="g", round=1, question="q"),), frame)
    assert not emission.programs
    assert any("shaped like a program" in note for note in emission.notes)


# -- against the real contract ---------------------------------------------


def test_emission_validates_against_labradors_own_contracts(record: RunResult, frame: ProgramFrame):
    """The only test that proves the handoff works. Everything above checks what
    this module intended; this checks what LABrador accepts."""
    models = pytest.importorskip(
        "labrador_roi.models",
        reason="install managed/program-strategy-valuation to run the contract test",
    )
    emission = emit(record, frame)
    assert emission.programs

    for payload in emission.programs:
        program = models.ProgramInput.model_validate(payload)
        assert program.patent.base_term_years == 20

        # The expansion shares the asset's clock: same filing year, no restart.
        for indication in [program.initial_indication, *program.expansion_indications]:
            assert indication.currency == program.currency

        # Nothing the graph supplied may support a critical screening input.
        for key, metadata in program.evidence.items():
            assert ":" in key
        for indication in [program.initial_indication, *program.expansion_indications]:
            assert not any(
                item.supports_decision for item in indication.population.evidence.values()
            )
            assert not any(item.supports_decision for item in indication.access.evidence.values())
