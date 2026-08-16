"""Each adapter as a program: reads documents, writes its artifact, nothing else.

These replace the flags that used to hang off the core CLI. The behaviours
under test are the same ones, moved: what a report must contain, that a
valuation refuses to invent an analyst's answer, and that everything is
recoverable from a saved document without re-running the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapters.report import cli as report_cli
from adapters.valuation import cli as valuation_cli
from adapters.webui import cli as webui_cli
from hyp_gen.cli import FILENAME
from hyp_gen.cli import main as core

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "examples" / "knowledge-graph.json"
FRAME = ROOT / "examples" / "analyst-frame.json"


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """One core run, on disk, as an adapter finds it."""
    out = tmp_path / "run"
    assert core(["--graph", str(GRAPH), "--dry-run", "--out", str(out)]) == 0
    return out


@pytest.fixture
def valuation_run(tmp_path: Path) -> Path:
    out = tmp_path / "vrun"
    assert core(["--graph", str(GRAPH), "--profile", "valuation", "--dry-run",
                 "--out", str(out)]) == 0
    return out


# -- report -----------------------------------------------------------------


def test_report_renders_from_a_saved_document_alone(run: Path, tmp_path: Path) -> None:
    """No graph, no key, no pipeline: the document is enough or it is not canonical."""
    out = tmp_path / "reports"
    assert report_cli.main([str(run / FILENAME), "--out", str(out)]) == 0

    rendered = (out / "report.md").read_text()
    assert rendered.startswith("# g_demo1 · round")


def test_every_mode_can_be_written_in_one_pass(run: Path, tmp_path: Path) -> None:
    out = tmp_path / "reports"
    assert report_cli.main([
        str(run / FILENAME), "--out", str(out),
        "--mode", "prose", "--mode", "table", "--mode", "trace", "--mode", "full",
    ]) == 0
    for name in ("report.md", "report-table.md", "report-trace.md", "report-full.md"):
        rendered = (out / name).read_text()
        assert rendered.startswith("# "), name
        assert "g_demo1" in rendered, name


def test_the_full_report_is_recoverable_later(run: Path, tmp_path: Path) -> None:
    """The short default is only safe because the long one costs nothing to get back."""
    out = tmp_path / "reports"
    assert report_cli.main([str(run / FILENAME), "--mode", "prose", "--out", str(out)]) == 0
    assert report_cli.main([str(run / FILENAME), "--mode", "full", "--out", str(out)]) == 0

    full = (out / "report-full.md").read_text()
    assert "**Scores**" in full
    assert len(full) > len((out / "report.md").read_text())


def test_an_unknown_mode_is_rejected(run: Path) -> None:
    with pytest.raises(SystemExit):
        report_cli.main([str(run / FILENAME), "--mode", "verbose"])


def test_a_directory_of_documents_is_read_as_one_bundle(run: Path, tmp_path: Path) -> None:
    out = tmp_path / "reports"
    assert report_cli.main([str(run), "--mode", "table", "--out", str(out)]) == 0
    assert "g_demo1" in (out / "report-table.md").read_text()


# -- webui ------------------------------------------------------------------


def test_webui_writes_cards_and_svg(run: Path, tmp_path: Path) -> None:
    cards, svg = tmp_path / "cards.json", tmp_path / "traces.svg"
    assert webui_cli.main([str(run / FILENAME), "--cards", str(cards), "--svg", str(svg)]) == 0

    payload = json.loads(cards.read_text())
    assert payload["hypotheses"], "a payload with no cards is not a payload"
    assert svg.read_text().startswith("<svg")


def test_webui_needs_to_be_told_what_to_write(run: Path) -> None:
    with pytest.raises(SystemExit):
        webui_cli.main([str(run / FILENAME)])


# -- valuation --------------------------------------------------------------


def test_valuation_refuses_without_a_frame(valuation_run: Path, tmp_path: Path, capsys) -> None:
    """The refusal is the feature. A default filing year would look sourced."""
    code = valuation_cli.main([str(valuation_run / FILENAME), "--out", str(tmp_path / "p")])
    assert code == 2
    assert "will not guess" in capsys.readouterr().err


def test_the_frame_template_is_written_and_is_not_yet_usable(
    valuation_run: Path, tmp_path: Path, capsys
) -> None:
    target = tmp_path / "frame.json"
    assert valuation_cli.main(["--emit-frame-template", str(target)]) == 0

    template = json.loads(target.read_text())
    assert template["filing_year"] is None

    assert valuation_cli.main([
        str(valuation_run / FILENAME), "--frame", str(target), "--out", str(tmp_path / "p"),
    ]) == 2
    assert "analyst" in capsys.readouterr().err


def test_valuation_writes_briefs_and_an_empty_catalogue(
    valuation_run: Path, tmp_path: Path
) -> None:
    out = tmp_path / "p"
    assert valuation_cli.main([
        str(valuation_run / FILENAME), "--frame", str(FRAME), "--out", str(out),
    ]) == 0

    assert sorted(out.glob("*.program.json"))
    # An empty catalogue rather than none: it makes the ROI model's missing-anchor
    # warning fire instead of hiding that no price was ever supplied.
    assert json.loads((out / "comparables.json").read_text()) == []
    assert json.loads((out / "emission.json").read_text())["graph_id"] == "g_demo1"
