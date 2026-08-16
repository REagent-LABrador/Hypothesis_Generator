from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from adapters.headless.cli import main
from adapters.headless.runner import run

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


def test_shared_contracts_are_exactly_pinned_and_cards_conform() -> None:
    lock = json.loads((ROOT / "contracts" / "contract.lock.json").read_text())
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


def test_complete_request_validates_against_current_roi_module_contract() -> None:
    contracts = pytest.importorskip(
        "labrador_roi.contracts",
        reason="install rnpv-roi-calculator to run the cross-repo contract test",
    )
    response = run(request(), mode="REPLAY")
    assert response.roi_request is not None
    validated = contracts.ModuleRunRequest.model_validate(response.roi_request)
    assert validated.request_id == "g_demo1-t8-roi"


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


def test_downstream_failure_preserves_the_canonical_hypothesis_and_cards() -> None:
    payload = request()
    payload["focus_thing_id"] = "t3"
    payload["profile"] = "mechanism"
    response = run(payload, mode="REPLAY")
    assert response.status == "CANNOT_COMPLETE"
    assert response.error is not None
    assert response.error.reason_code == "ROI_PROGRAM_NOT_EMITTED"
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
