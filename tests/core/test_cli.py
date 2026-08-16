"""The core command line: parameter patching, the keyless path, failure modes.

What this suite is *not* testing is as deliberate as what it is: reports, UI
payloads and program briefs are not flags on this program any more. They have
their own CLIs, and their own tests under tests/adapters/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyp_gen.cli import FILENAME, _overrides, main

GRAPH = Path(__file__).resolve().parents[2] / "examples" / "knowledge-graph.json"


def test_overrides_parse_json_values() -> None:
    parsed = _overrides(
        ["traversal.max_hops=4", "framing.mode=closed", "loop.enabled=true",
         'framing.anchors=["metformin"]']
    )
    assert parsed["traversal"]["max_hops"] == 4
    assert parsed["framing"]["mode"] == "closed"      # bare strings stay strings
    assert parsed["loop"]["enabled"] is True
    assert parsed["framing"]["anchors"] == ["metformin"]


def test_overrides_reject_malformed_pairs() -> None:
    with pytest.raises(SystemExit):
        _overrides(["max_hops=4"])  # no group


def test_dry_run_needs_no_credentials(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = main(["--graph", str(GRAPH), "--dry-run", "--out", str(tmp_path)])
    assert code == 0

    # Diagnostics on stderr, so stdout stays parseable.
    assert "No model calls made." in capsys.readouterr().err

    document = json.loads((tmp_path / FILENAME).read_text())
    assert document["provenance"]["graph_id"] == "g_demo1"
    assert document["provenance"]["counts"]["model_calls"] == 0


def test_the_run_says_what_it_chose_and_from_how_many(tmp_path: Path, capsys) -> None:
    """One hypothesis out of eight is a different claim from one out of one."""
    assert main(["--graph", str(GRAPH), "--dry-run", "--out", str(tmp_path)]) == 0
    err = capsys.readouterr().err
    assert "chosen from" in err
    assert FILENAME in err


def test_without_out_the_document_goes_to_stdout(tmp_path: Path, capsys, monkeypatch) -> None:
    """A pipeable core is what makes an adapter a separate program rather than a flag."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--graph", str(GRAPH), "--dry-run"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["hypothesis"]["id"], "stdout must be the document and nothing else"


def test_missing_credentials_fail_clearly(tmp_path: Path, capsys, monkeypatch) -> None:
    """A stack trace forty frames deep is not an error message."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    code = main(["--graph", str(GRAPH), "--out", str(tmp_path)])
    assert code == 2

    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "--dry-run" in err
    assert not (tmp_path / FILENAME).exists()


def test_profile_and_set_compose(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = main(
        [
            "--graph", str(GRAPH),
            "--profile", "repurposing",
            "--set", "selection.top_k=2",
            "--dry-run",
            "--out", str(tmp_path),
        ]
    )
    assert code == 0
    provenance = json.loads((tmp_path / FILENAME).read_text())["provenance"]
    # The patch applied on top of the profile, and the profile survived it.
    assert provenance["params"]["selection"]["top_k"] == 2
    assert provenance["params"]["traversal"]["seed_kinds"] == ["small_molecule"]
    assert provenance["considered"] == 2


def test_params_travel_with_the_document(tmp_path: Path, monkeypatch) -> None:
    """A hypothesis whose parameters are not attached cannot be reproduced."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    main(["--graph", str(GRAPH), "--dry-run", "--out", str(tmp_path)])
    params = json.loads((tmp_path / FILENAME).read_text())["provenance"]["params"]
    for group in ("framing", "traversal", "motifs", "evidence", "novelty",
                  "selection", "ranking", "loop", "budget"):
        assert group in params


def test_the_core_has_no_adapter_flags(capsys) -> None:
    """Adapters are programs, not options. Their flags must not creep back."""
    with pytest.raises(SystemExit):
        main(["--graph", str(GRAPH), "--dry-run", "--report-mode", "full"])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_focus_is_recorded_and_the_winner_contains_it(tmp_path: Path) -> None:
    assert main([
        "--graph", str(GRAPH), "--profile", "valuation", "--focus-thing-id", "t8",
        "--dry-run", "--out", str(tmp_path),
    ]) == 0
    document = json.loads((tmp_path / FILENAME).read_text())
    assert document["provenance"]["params"]["focus_thing_id"] == "t8"
    hypothesis = document["hypothesis"]
    assert "t8" in {
        hypothesis["subject"],
        hypothesis["object"],
        *(step["from"] for step in hypothesis["path"]),
        *(step["to"] for step in hypothesis["path"]),
    }


def test_unknown_focus_is_rejected_before_running(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit):
        main([
            "--graph", str(GRAPH), "--focus-thing-id", "not-a-node",
            "--dry-run", "--out", str(tmp_path),
        ])
    assert "not present" in capsys.readouterr().err
