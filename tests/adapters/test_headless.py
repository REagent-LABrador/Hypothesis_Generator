from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from adapters.headless.cli import main
from adapters.headless.runner import _scope_document, run
from hyp_gen.hypothesis import Ask

ROOT = Path(__file__).resolve().parents[2]


def request() -> dict:
    frame = json.loads((ROOT / "examples" / "analyst-frame.json").read_text())
    frame.pop("_README")
    return {
        "graph": json.loads((ROOT / "examples" / "knowledge-graph.json").read_text()),
        "focus_thing_id": "t8",
        "profile": "valuation",
        "valuation_frame": frame,
        "roi": {
            "request_id": "g_demo1-t8-roi",
            "comparables": [],
            "execution": {
                "simulations": 128,
                "seed": 42,
                "simulation_assumptions": {},
            },
        },
    }


def golden_request(focus_thing_id: str) -> dict:
    return {
        "graph": json.loads(
            (ROOT / "tests" / "fixtures" / "ra-irak4-evidence.json").read_text()
        ),
        "focus_thing_id": focus_thing_id,
        "profile": "default",
        "valuation_frame": {
            "base_year": 2026,
            "valuation_year": 2026,
            "launch_year": 2034,
            "filing_year": 2026,
            "currency": "USD",
            "geography": "United States",
            "therapeutic_area": "Immunology",
            "target_population": "Adults with active rheumatoid arthritis",
            "line_of_therapy": "Second line",
            "route": "ORAL",
            "current_stage": "preclinical",
            "modality": "SMALL_MOLECULE",
            "target": "IRAK4",
            "expansion_launch_year": None,
            "notes": (
                "Analyst-supplied valuation assumptions for focused interoperability "
                "testing; not graph findings."
            ),
        },
        "roi": {
            "request_id": f"IRAK4-RA-{focus_thing_id}",
            "comparables": [],
            "execution": {
                "simulations": 128,
                "seed": 42,
                "simulation_assumptions": {},
            },
        },
    }


def test_shared_contracts_are_exactly_pinned_and_cards_conform() -> None:
    for lock_name in ("contract.lock.json", "roi-contract.lock.json"):
        lock = json.loads((ROOT / "contracts" / lock_name).read_text())
        for path, metadata in lock["files"].items():
            content = (ROOT / path).read_bytes()
            assert hashlib.sha256(content).hexdigest() == metadata["sha256"]

    response = run(request(), mode="REPLAY")
    assert response.cards is not None
    schema = json.loads((ROOT / "contracts" / "interpretability.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(
        response.cards.interpretability.model_dump(mode="json")
    )
    thesis = json.loads((ROOT / "contracts" / "indication-thesis.schema.json").read_text())
    Draft202012Validator.check_schema(thesis)


def test_replay_is_provider_free_and_returns_the_full_handoff() -> None:
    def provider_must_not_be_created(**_kwargs):
        raise AssertionError("REPLAY instantiated the provider-backed Judge")

    response = run(
        request(),
        mode="REPLAY",
        judge_factory=provider_must_not_be_created,
    )
    assert response.status == "COMPLETE"
    assert response.execution_mode == "REPLAY"
    assert response.output_origin == "DETERMINISTIC_REPLAY"
    assert response.error is None
    assert response.hypothesis is not None
    assert response.hypothesis.provenance.params["focus_thing_id"] == "t8"
    assert "t8" in response.hypothesis.hypothesis.evidence["things"]
    assert response.cards is not None and response.cards.interpretability
    assert response.cards.hypotheses[0].id == response.hypothesis.hypothesis.id
    assert response.roi_request is not None
    assert response.roi_request.keys() == {
        "contract_version",
        "module",
        "request_id",
        "program",
        "comparables",
        "execution",
    }
    assert response.roi_request["module"] == "rnpv_roi_calculator"
    assert response.roi_request["request_id"] == "g_demo1-t8-roi"
    assert response.roi_request["execution"]["seed"] == 42


def test_headless_request_and_response_match_the_committed_schemas() -> None:
    payload = request()
    response = run(payload, mode="REPLAY")
    input_schema = json.loads((ROOT / "schemas" / "headless-input.schema.json").read_text())
    output_schema = json.loads((ROOT / "schemas" / "headless-output.schema.json").read_text())

    Draft202012Validator.check_schema(input_schema)
    Draft202012Validator.check_schema(output_schema)
    Draft202012Validator(input_schema).validate(payload)
    Draft202012Validator(output_schema).validate(response.model_dump(mode="json"))


def test_complete_request_validates_against_current_roi_module_contract() -> None:
    contracts = pytest.importorskip(
        "labrador_roi.contracts",
        reason="install rnpv-roi-calculator to run the cross-repo contract test",
    )
    response = run(request(), mode="REPLAY")
    assert response.roi_request is not None
    validated = contracts.ModuleRunRequest.model_validate(response.roi_request)
    assert validated.request_id == "g_demo1-t8-roi"


def test_golden_process_focuses_are_distinct_complete_roi_handoffs() -> None:
    hypothesis_schema = json.loads(
        (ROOT / "schemas" / "hypothesis.schema.json").read_text()
    )
    cards_schema = json.loads((ROOT / "schemas" / "cards.schema.json").read_text())
    input_schema = json.loads((ROOT / "schemas" / "headless-input.schema.json").read_text())
    output_schema = json.loads(
        (ROOT / "schemas" / "headless-output.schema.json").read_text()
    )
    roi_schema = json.loads((ROOT / "contracts" / "roi-input.schema.json").read_text())
    for schema in (
        hypothesis_schema,
        cards_schema,
        input_schema,
        output_schema,
        roi_schema,
    ):
        Draft202012Validator.check_schema(schema)

    focused_ids: set[str] = set()
    program_ids: set[str] = set()
    for focus_thing_id in ("t2", "t3", "t5"):
        payload = golden_request(focus_thing_id)
        response = run(payload, mode="REPLAY")
        repeated = run(payload, mode="REPLAY")

        assert response.status == "COMPLETE"
        assert response.error is None
        assert response.hypothesis is not None
        assert response.cards is not None
        assert response.roi_request is not None

        hypothesis_id = response.hypothesis.hypothesis.id
        assert repeated.hypothesis is not None
        assert repeated.hypothesis.hypothesis.id == hypothesis_id
        assert response.hypothesis.provenance.params["focused_identity"] == {
            "base_hypothesis_id": hypothesis_id.split("--focus-", 1)[0],
            "focus_thing_id": focus_thing_id,
        }
        assert response.cards.hypotheses[0].id == hypothesis_id
        assert response.cards.interpretability.headline.result == hypothesis_id
        for ask in [
            *response.hypothesis.asks,
            *response.hypothesis.hypothesis.asks,
        ]:
            assert ask.for_hypothesis in (None, hypothesis_id)

        program = response.roi_request["program"]
        indication = program["initial_indication"]
        assert program["assumptions"]["hypothesis_ids"] == [hypothesis_id]
        assert program["assumptions"]["focused_hypothesis_id"] == hypothesis_id
        assert program["assumptions"]["focused_hypothesis_mechanism"]
        assert program["assumptions"]["economic_indication_source"] == (
            "valuation_frame.target_population"
        )
        assert "not a discovered or nominated molecule" in program["assumptions"][
            "program_name_limitation"
        ]
        assert "not a discovered or nominated molecule" in program["assumptions"][
            "molecule_identifier_limitation"
        ]
        assert program["program_name"] == "IRAK4 in Adults with active rheumatoid arthritis"
        assert program["molecule_identifier"] == "frame-target:IRAK4:SMALL_MOLECULE"
        assert indication["name"] == payload["valuation_frame"]["target_population"]
        assert indication["target_population"] == payload["valuation_frame"][
            "target_population"
        ]
        assert indication["assumptions"]["hypothesis_id"] == hypothesis_id
        assert indication["assumptions"]["indication_identity_source"] == (
            "valuation_frame.target_population"
        )
        assert f"hypothesis:{hypothesis_id}" in indication["evidence"]

        # A complete request is an executable gap analysis, not permission to
        # invent payer, epidemiology, price, or development inputs.
        assert indication["population"]["eligible_patients"] is None
        assert indication["access"]["coverage_fraction"] is None
        assert indication["access"]["adoption_by_year"] == {}
        assert program["development"]["stage_costs"] == {}
        assert program["development"]["program_probability_of_approval"] is None

        Draft202012Validator(input_schema).validate(payload)
        Draft202012Validator(hypothesis_schema).validate(
            response.hypothesis.model_dump(mode="json")
        )
        Draft202012Validator(cards_schema).validate(response.cards.model_dump(mode="json"))
        Draft202012Validator(output_schema).validate(response.model_dump(mode="json"))
        Draft202012Validator(roi_schema).validate(response.roi_request)

        focused_ids.add(hypothesis_id)
        program_ids.add(program["program_id"])

    assert len(focused_ids) == 3
    assert len(program_ids) == 3


def test_focus_scope_retargets_document_and_hypothesis_asks() -> None:
    response = run(request(), mode="REPLAY")
    assert response.hypothesis is not None
    old_id = "H-unscoped"
    ask = Ask(
        graph_id=response.hypothesis.provenance.graph_id,
        ask="test_gap",
        target="g-test",
        for_hypothesis=old_id,
    )
    unscoped = response.hypothesis.model_copy(
        deep=True,
        update={
            "hypothesis": response.hypothesis.hypothesis.model_copy(
                deep=True,
                update={"id": old_id, "asks": [ask]},
            ),
            "asks": [ask],
        },
    )

    scoped = _scope_document(unscoped, focus_thing_id="t8")
    assert scoped.hypothesis.id != old_id
    assert scoped.hypothesis.asks[0].for_hypothesis == scoped.hypothesis.id
    assert scoped.asks[0].for_hypothesis == scoped.hypothesis.id


def test_live_never_falls_back_when_credentials_are_missing() -> None:
    class NoCredentialJudge:
        def __init__(self, **_kwargs):
            pass

        def has_credentials(self) -> bool:
            return False

    response = run(request(), mode="LIVE", judge_factory=NoCredentialJudge)
    assert response.status == "CANNOT_COMPLETE"
    assert response.output_origin == "LIVE_PROVIDER"
    assert response.error is not None
    assert response.error.reason_code == "CREDENTIAL_MISSING"
    assert response.hypothesis is None


def test_incomplete_focused_frame_preserves_the_hypothesis_and_cards() -> None:
    payload = request()
    payload["focus_thing_id"] = "t3"
    payload["profile"] = "mechanism"
    response = run(payload, mode="REPLAY")
    assert response.status == "CANNOT_COMPLETE"
    assert response.error is not None
    assert response.error.reason_code == "ROI_FRAME_INCOMPLETE"
    assert "target, modality" in response.error.message
    assert response.hypothesis is not None
    assert response.cards is not None
    assert response.roi_request is None


def test_unknown_focus_has_an_explicit_terminal_reason() -> None:
    payload = request()
    payload["focus_thing_id"] = "t404"
    response = run(payload, mode="REPLAY")
    assert response.status == "CANNOT_COMPLETE"
    assert response.error is not None
    assert response.error.reason_code == "FOCUS_THING_NOT_FOUND"


def test_file_in_file_out_replay(tmp_path: Path) -> None:
    input_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(request()))
    assert main([
        "--mode", "replay", "--input", str(input_path), "--output", str(output_path)
    ]) == 0
    payload = json.loads(output_path.read_text())
    assert payload["status"] == "COMPLETE"
    assert payload["output_origin"] == "DETERMINISTIC_REPLAY"
    assert payload["hypothesis"]["hypothesis"]["id"]
    assert payload["cards"]["interpretability"]
    assert payload["roi_request"]["contract_version"] == "1.0.0"
