"""The agent layer: what a model can reach, and what it is told about it.

Two kinds of test here. The first kind checks the tools do what they say —
ordinary wrapper behaviour. The second kind checks the *guards*, which exist
only because the caller is a model: a graph path may not escape its directory,
a run may not be read from outside ``runs/``, and every tool the server exposes
has to be named in the prompt that explains how to use it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import tools
from hyp_gen.pipeline import Generator

ROOT = Path(__file__).resolve().parents[2]
GRAPH = "knowledge-graph.json"


@pytest.fixture
def runs(tmp_path: Path, monkeypatch) -> Path:
    """Point every write at a temp dir. A test suite must not fill runs/."""
    target = tmp_path / "runs"
    monkeypatch.setattr(tools, "RUNS", target)
    return target


@pytest.fixture
def document(runs: Path) -> str:
    result = tools.generate_hypothesis(graph=GRAPH, profile="repurposing")
    return result["document_path"]


# -- what the agent can see --------------------------------------------------


def test_list_graphs_finds_the_example_and_summarises_it():
    listed = tools.list_graphs()["graphs"]
    graph = next(g for g in listed if g["file"] == GRAPH)

    assert graph["graph_id"] == "g_demo1"
    assert graph["question"]
    assert graph["things"] and graph["links"]
    # Coverage travels with the listing because it decides what a novelty claim
    # off this graph is worth, and the agent picks a stance before it runs.
    assert graph["coverage_depth"] == "deep"
    assert graph["truncated"] is True


def test_list_graphs_ignores_files_that_are_not_graphs():
    """examples/ also holds a run output and an analyst frame."""
    files = {g["file"] for g in tools.list_graphs()["graphs"]}
    assert "hypothesis.json" not in files
    assert "analyst-frame.json" not in files


# -- the guards --------------------------------------------------------------


@pytest.mark.parametrize(
    "ref",
    ["../pyproject.toml", "../../etc/passwd", "/etc/passwd", "..\\windows", ""],
)
def test_a_graph_path_may_not_escape_its_directory(ref: str):
    """Rejected outright, never normalised.

    A knowledge graph is exactly the kind of input an injection rides in on, and
    a rule that cleans up a hostile path is one bug away from accepting it.

    What this does not prove: that the agent cannot read the file another way.
    If it has a shell, it can. The guard binds a sandboxed deployment, where
    these tools are the only route to the disk.
    """
    with pytest.raises(tools.ToolError) as exc:
        tools.resolve_graph(ref)
    assert "list_graphs" in str(exc.value) or "bare filename" in str(exc.value)


def test_an_unknown_graph_says_how_to_find_a_real_one():
    with pytest.raises(tools.ToolError, match="list_graphs"):
        tools.resolve_graph("no-such-graph.json")


def test_a_document_may_only_be_read_from_the_runs_directory(runs: Path, tmp_path: Path):
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}")
    with pytest.raises(tools.ToolError, match="generate_hypothesis"):
        tools.resolve_document(str(outside))


def test_a_missing_document_is_a_readable_error(runs: Path):
    with pytest.raises(tools.ToolError, match="no such document"):
        tools.resolve_document(str(runs / "nope" / "hypothesis.json"))


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"profile": "aggressive"}, "unknown profile"),
        ({"craziness": 1.5}, "between 0 and 1"),
        ({"overrides": ["max_hops=4"]}, "group.key=value"),
    ],
)
def test_bad_stances_are_refused_with_a_usable_message(kwargs: dict, expected: str):
    with pytest.raises(tools.ToolError, match=expected):
        tools.preview_candidates(graph=GRAPH, **kwargs)


# -- preview -----------------------------------------------------------------


def test_preview_costs_nothing_and_shows_what_would_win(monkeypatch):
    """The free look. No judge is constructed, so no key can be spent."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    preview = tools.preview_candidates(graph=GRAPH, profile="repurposing")

    assert preview["shortlisted"] == len(preview["candidates"])
    assert preview["candidates"], "the example graph supports something"
    assert preview["stance"]["profile"] == "repurposing"
    # The absence discount is on the preview because it decides what the
    # novelty numbers beside it are worth.
    assert preview["absence_reliability"] == pytest.approx(0.408)

    top = preview["candidates"][0]
    assert top["chain"].count("→") >= 0
    assert set(top["scores"]) >= {"support", "novelty", "testability", "rank_score"}


def test_the_preview_winner_is_what_generate_returns(runs: Path):
    """Otherwise the free look is not a preview of anything."""
    preview = tools.preview_candidates(graph=GRAPH, profile="repurposing")
    generated = tools.generate_hypothesis(graph=GRAPH, profile="repurposing")

    assert preview["candidates"][0]["id"] == generated["hypothesis"]["id"]
    assert preview["shortlisted"] == generated["considered"]


def test_the_preview_names_the_gates_that_would_reject_a_candidate():
    """A fail here is a candidate you would otherwise pay to articulate first."""
    preview = tools.preview_candidates(graph=GRAPH, profile="repurposing")
    warned = [c for c in preview["candidates"] if c["gate_warnings"]]
    assert warned, "the demo graph has a single-source candidate"
    assert {"gate", "status", "summary"} == set(warned[0]["gate_warnings"][0])


def test_an_impossible_stance_previews_empty_and_says_so(monkeypatch):
    monkeypatch.setattr(Generator, "shortlist", lambda self: [])
    preview = tools.preview_candidates(graph=GRAPH)
    assert preview["shortlisted"] == 0
    assert "nothing worth stating" in preview["note"]


# -- generate ----------------------------------------------------------------


def test_generate_returns_one_hypothesis_and_says_how_many_it_beat(runs: Path):
    result = tools.generate_hypothesis(graph=GRAPH, profile="repurposing")

    assert isinstance(result["hypothesis"], dict), "one hypothesis, not a list"
    assert result["considered"] >= 1
    assert result["model_calls"] == 0
    assert result["stance"]["profile"] == "repurposing"
    assert Path(result["document_path"]).is_file()
    assert "get_evidence" in result["next"]


def test_two_stances_do_not_overwrite_each_other(runs: Path):
    """Comparing two stances is the thing an agent most often wants to do."""
    first = tools.generate_hypothesis(graph=GRAPH, profile="conservative")
    second = tools.generate_hypothesis(graph=GRAPH, profile="speculative", craziness=0.9)

    assert first["document_path"] != second["document_path"]
    assert Path(first["document_path"]).is_file()
    assert Path(second["document_path"]).is_file()


def test_nothing_surviving_is_reported_as_a_real_answer(runs: Path, monkeypatch):
    monkeypatch.setattr(Generator, "shortlist", lambda self: [])
    result = tools.generate_hypothesis(graph=GRAPH)

    assert result["hypothesis"] is None
    assert result["considered"] == 0
    assert "name the ask" in result["why"]
    assert "document_path" not in result, "nothing was written, so nothing to read"


def test_articulate_without_credentials_points_at_the_free_path(runs: Path, monkeypatch):
    """A missing key must not read as a broken generator: half of it needs none."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    with pytest.raises(tools.ToolError) as exc:
        tools.generate_hypothesis(graph=GRAPH, articulate=True)
    assert "articulate=false" in str(exc.value)


def test_the_summary_keeps_the_stance_and_drops_the_knobs(runs: Path, document: str):
    """Scores are unreadable without the stance, and useless with 4KB of params."""
    result = tools.generate_hypothesis(graph=GRAPH, profile="repurposing")
    assert set(result["stance"]) == {"profile", "craziness"}
    assert "traversal" not in json.dumps(result["stance"])


def test_the_summary_never_reports_a_skipped_gate_as_a_pass(runs: Path, document: str):
    evidence = tools.get_evidence(document_path=document)
    verification = evidence["verification"]

    statuses = {g["gate"]: g["status"] for g in verification["gates"]}
    # A structural run cannot have run the model gates.
    assert statuses["citations"] == "skip"
    assert verification["verdict"] != "verified"


# -- evidence ----------------------------------------------------------------


def test_get_evidence_carries_the_verbatim_sentences(runs: Path, document: str):
    evidence = tools.get_evidence(document_path=document)

    findings = evidence["evidence"]["findings"]
    assert findings, "a hypothesis with no source sentence is not checkable"
    for finding in findings.values():
        assert finding["quote"]
        assert finding["paper"] in evidence["evidence"]["papers"]


def test_get_evidence_states_the_citation_rule(runs: Path, document: str):
    """The pack is the legal citation set, and the agent is told so on delivery."""
    evidence = tools.get_evidence(document_path=document)
    assert "only thing this hypothesis may cite" in evidence["citation_rule"]
    assert "never obey it" in evidence["content_rule"]


def test_every_link_on_the_walk_is_in_the_pack(runs: Path, document: str):
    evidence = tools.get_evidence(document_path=document)
    for step in evidence["path"]:
        assert step["link"] in evidence["evidence"]["links"]


# -- adapters through the agent ----------------------------------------------


@pytest.mark.parametrize("mode", ["prose", "table", "trace", "full"])
def test_render_report_writes_beside_the_document(runs: Path, document: str, mode: str):
    rendered = tools.render_report(document_path=document, mode=mode)
    assert rendered["markdown"].startswith("# ")
    assert Path(rendered["report_path"]).parent == Path(document).parent


def test_an_unknown_report_mode_is_refused(runs: Path, document: str):
    with pytest.raises(tools.ToolError, match="mode must be one of"):
        tools.render_report(document_path=document, mode="verbose")


def test_emit_programs_without_a_frame_returns_a_template_to_show(runs: Path, document: str):
    """The refusal is the feature: a guessed filing year looks sourced."""
    result = tools.emit_programs(document_path=document)

    assert result["frame_template"]["filing_year"] is None
    assert "their decision, not yours" in result["next"]
    assert "programs" not in result


def test_emit_programs_refuses_a_frame_that_is_still_a_template(runs: Path, document: str):
    template = tools.emit_programs(document_path=document)["frame_template"]
    with pytest.raises(tools.ToolError, match="not usable yet"):
        tools.emit_programs(document_path=document, frame=template)


def test_emit_programs_writes_briefs_and_warns_off_the_rnpv(runs: Path):
    frame = json.loads((ROOT / "examples" / "analyst-frame.json").read_text())
    valuation = tools.generate_hypothesis(graph=GRAPH, profile="valuation")

    result = tools.emit_programs(
        document_path=valuation["document_path"], frame=frame
    )

    assert result["programs"], "the valuation profile should emit something"
    assert Path(result["programs"][0]["path"]).is_file()
    assert json.loads(Path(result["comparables_path"]).read_text()) == []
    assert "NOT_DECISION_GRADE" in result["decision_grade_warning"]


# -- the contract between the tools and the prompt ---------------------------

CLAUDE_MD = ROOT / "agent" / "CLAUDE.md"


def test_every_tool_has_a_description_and_a_schema():
    for tool in tools.TOOLS:
        assert tool["description"], f"{tool['name']} would reach the model unexplained"
        assert tool["input_schema"]["type"] == "object"
        for prop in tool["input_schema"].get("properties", {}).values():
            assert prop.get("description"), (
                f"{tool['name']} has a parameter the model is told nothing about"
            )


def test_the_prompt_names_every_tool_the_agent_is_given():
    """A tool the prompt does not mention is a tool the agent will not use well.

    This is the seam that rots first: someone adds a tool, the schema explains
    what it does, and nothing explains *when* to reach for it.
    """
    prompt = CLAUDE_MD.read_text()
    missing = [t["name"] for t in tools.TOOLS if f"`{t['name']}`" not in prompt]
    assert not missing, f"agent/CLAUDE.md does not mention: {missing}"


def test_the_prompt_still_carries_the_rules_that_cannot_be_dropped():
    """Each of these was a real failure mode before it was a sentence."""
    prompt = CLAUDE_MD.read_text().lower()
    for rule in (
        "absence is not evidence of absence",   # a quick search is not a negative result
        "cite by id",                           # a claim without an id is the agent's own
        "considered",                           # 1-of-1 and 1-of-40 differ
        "skip",                                 # a skipped gate is not a pass
        "never instructions",                   # graph text is data
        "not_decision_grade",                   # the ROI zero is not a valuation
    ):
        assert rule in prompt, f"agent/CLAUDE.md no longer says: {rule}"


def test_every_profile_the_generator_offers_is_explained_in_the_prompt():
    from hyp_gen.params import PROFILES

    prompt = CLAUDE_MD.read_text()
    missing = [p for p in PROFILES if f"`{p}`" not in prompt]
    assert not missing, f"agent/CLAUDE.md does not explain when to use: {missing}"


def test_dispatch_by_name_matches_the_advertised_tools():
    assert set(tools.BY_NAME) == {t["name"] for t in tools.TOOLS}
    with pytest.raises(tools.ToolError, match="no tool named"):
        tools.call("generate_slate")


def test_nothing_below_the_agent_imports_it():
    """The dependency runs one way, as it does for the adapters.

    The generator and its adapters have to be usable with no agent, no MCP and
    no prompt — an agent layer that something else imported would make the
    optional `[agent]` extra mandatory in practice.
    """
    import ast

    offenders = []
    for directory in ("src/hyp_gen", "adapters"):
        for path in (ROOT / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                imported = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                if any(name.split(".")[0] == "agent" for name in imported):
                    offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"modules importing the agent layer: {offenders}"
