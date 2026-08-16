"""The app's boundary: one graph in, ONE hypothesis out.

Everything here guards a promise made to someone outside this codebase, so a
failure means a consumer breaks -- not that an internal detail moved.
"""

from __future__ import annotations

import json
from pathlib import Path

import generate_schemas  # tools/, on the path via pyproject's pytest pythonpath
import pytest

from hyp_gen.cli import FILENAME, main
from hyp_gen.graph import KnowledgeGraph
from hyp_gen.hypothesis import SCHEMA_VERSION, HypothesisDocument
from hyp_gen.params import Params
from hyp_gen.pipeline import Generator

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "examples" / "knowledge-graph.json"


def test_every_committed_schema_matches_the_code():
    """A stale schema is worse than none: it documents a contract nobody honours."""
    stale = [
        path.relative_to(ROOT)
        for path, text in generate_schemas.documents()
        if not path.exists() or path.read_text() != text
    ]
    stale += [
        path.relative_to(ROOT)
        for path, schema in generate_schemas.json_schemas()
        if not path.exists()
        or path.read_text() != json.dumps(schema, indent=2, sort_keys=True) + "\n"
    ]
    assert not stale, (
        f"stale schema files: {stale} -- run `python tools/generate_schemas.py` "
        "and commit the result"
    )


SCHEMA = ROOT / "schemas" / "SCHEMA.md"


def test_the_core_and_every_adapter_have_one_contract_each():
    """Anything that produces an artifact says what it consumes and what it emits."""
    assert SCHEMA.exists(), "the core contract is schemas/SCHEMA.md"
    for adapter in sorted((ROOT / "adapters").glob("*/")):
        if adapter.name.startswith((".", "_")):
            continue
        assert (adapter / "SCHEMA.md").exists(), f"{adapter.name} has no SCHEMA.md"


def _fields(model, seen=None) -> set[str]:
    """Every field name in a model and everything it nests."""
    from pydantic import BaseModel

    seen = seen if seen is not None else set()
    names: set[str] = set()
    for name, field in model.model_fields.items():
        names.add(field.alias or name)
        for arg in (field.annotation, *getattr(field.annotation, "__args__", ())):
            for inner in (arg, *getattr(arg, "__args__", ())):
                if (
                    isinstance(inner, type)
                    and issubclass(inner, BaseModel)
                    and inner not in seen
                ):
                    seen.add(inner)
                    names |= _fields(inner, seen)
    return names


def test_the_contract_documents_every_field_of_both_models():
    """SCHEMA.md is written by hand, so this is what stops it falling behind.

    A generated field table cannot say what `searched_in_round: null` means or
    why `path` holds the donor's edge, which is why the core's contract is
    authored -- but a human forgets to add the field they just shipped, and a
    contract missing a field is worse than one that never mentioned it.
    """
    documented = SCHEMA.read_text()
    missing = sorted(
        name
        for name in _fields(KnowledgeGraph) | _fields(HypothesisDocument)
        if f"\"{name}\"" not in documented and f"`{name}`" not in documented
    )
    assert not missing, (
        f"schemas/SCHEMA.md does not mention: {missing}. A field a consumer will "
        "see in the JSON has to appear in the contract that describes it."
    )


def test_the_worked_example_is_a_real_current_run():
    """The example in SCHEMA.md is copied from this file, so it must not go stale."""
    committed = json.loads((ROOT / "examples" / "hypothesis.json").read_text())
    HypothesisDocument.model_validate(committed)

    fresh = Generator(
        KnowledgeGraph.load(GRAPH), Params.profile("repurposing")
    ).run().top()
    assert fresh.hypothesis.id == committed["hypothesis"]["id"], (
        "examples/hypothesis.json no longer matches what the code produces -- "
        "rerun `hypgen --graph examples/knowledge-graph.json --profile repurposing "
        "--dry-run --out examples/` and update the worked example in SCHEMA.md"
    )
    assert fresh.provenance.considered == committed["provenance"]["considered"]


def test_the_example_graph_satisfies_the_input_schema():
    graph = KnowledgeGraph.load(GRAPH)
    assert graph.things and graph.links, "the shipped example must exercise the contract"


def test_a_run_writes_one_hypothesis_and_calls_it_that(tmp_path):
    out = tmp_path / "run"
    assert main(["--graph", str(GRAPH), "--dry-run", "--out", str(out)]) == 0

    written = list(out.glob("*.json"))
    assert [p.name for p in written] == [FILENAME], (
        "a run writes exactly one document; a slate is not the output of this app"
    )

    payload = json.loads(written[0].read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    document = HypothesisDocument.model_validate(payload)

    # A single hypothesis, not a list of them, at the top level.
    assert isinstance(payload["hypothesis"], dict)
    assert "hypotheses" not in payload
    assert document.hypothesis.id


def test_the_document_carries_the_provenance_needed_to_judge_it(tmp_path):
    """A hypothesis away from its run is unfalsifiable in practice.

    Support 0.5 at craziness 0.1 and support 0.5 at craziness 0.9 are different
    claims about the world, and the score cannot tell them apart.
    """
    out = tmp_path / "run"
    assert main(["--graph", str(GRAPH), "--dry-run", "--craziness", "0.9", "--out", str(out)]) == 0
    document = HypothesisDocument.model_validate_json((out / FILENAME).read_text())

    p = document.provenance
    assert p.graph_id and p.question
    assert p.coverage, "coverage decides whether novelty means anything"
    assert p.params["stance"]["craziness"] == pytest.approx(0.9)
    assert p.considered >= 1, "a reader should know selection happened"


def test_the_winner_is_the_top_ranked_candidate():
    """Selection still runs; only its winner crosses the boundary."""
    result = Generator(KnowledgeGraph.load(GRAPH), Params()).run()
    document = result.top()

    assert len(result.hypotheses) > 1, "the example graph should support a real choice"
    assert document.hypothesis.id == result.hypotheses[0].id
    assert document.provenance.considered == len(result.hypotheses)


def test_asks_travel_with_the_hypothesis_they_belong_to():
    result = Generator(KnowledgeGraph.load(GRAPH), Params()).run()
    document = result.top()
    for ask in document.asks:
        assert ask.for_hypothesis in (None, document.hypothesis.id), (
            "a document may not carry another hypothesis's next step"
        )


def test_nothing_is_written_when_nothing_survives(tmp_path, monkeypatch):
    """An empty answer is a real answer, and better than promoting the least bad row."""
    monkeypatch.setattr(Generator, "shortlist", lambda self: [])
    out = tmp_path / "run"
    assert main(["--graph", str(GRAPH), "--dry-run", "--out", str(out)]) == 1
    assert not (out / FILENAME).exists()


def test_the_core_never_imports_an_adapter():
    """The dependency runs one way. An adapter is optional by construction."""
    import ast

    offenders = []
    for path in (ROOT / "src" / "hyp_gen").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            imported = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(name.split(".")[0] == "adapters" for name in imported):
                offenders.append(path.relative_to(ROOT))
    assert not offenders, f"core modules referencing adapters: {offenders}"
