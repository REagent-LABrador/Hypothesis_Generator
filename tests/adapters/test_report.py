"""The report adapter: hypothesis documents in, markdown out.

These tests moved here from the pipeline suite when reports stopped being the
core's job. What they check has not changed: a mode may change the form, never
the safety.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.report import render as report
from adapters.report.render import to_markdown
from hyp_gen.graph import KnowledgeGraph
from hyp_gen.hypothesis import (
    Articulation,
    Claim,
    Critique,
    GateResult,
    Hypothesis,
    ValidationIssue,
    Verification,
)
from hyp_gen.params import EvidenceParams, Params, RankingParams, SelectionParams
from conftest import bundle
from fakes import FakeJudge, _params
from hyp_gen.pipeline import Generator, RunResult

GRAPH = Path(__file__).resolve().parents[2] / "examples" / "knowledge-graph.json"


def test_report_renders_the_audit_trail(graph: KnowledgeGraph) -> None:
    record = bundle(Generator(graph=graph, params=_params(), judge=FakeJudge()).run())
    markdown = to_markdown(record, mode="full")
    assert "# Hypotheses" in markdown
    assert "Killed by" in markdown
    assert "Source sentences" in markdown
    # The coverage warning is not optional on a truncated graph.
    assert "not** evidence of absence" in markdown


def test_the_brief_report_is_the_default_and_is_much_shorter(
    graph: KnowledgeGraph,
) -> None:
    """report.md is read by humans; the audit trail is read by auditors.

    Brief being the *default* is the point -- a reader who has to know about a
    flag to get a readable report does not get one.
    """
    record = bundle(Generator(graph=graph, params=_params(), judge=FakeJudge()).run())
    brief = to_markdown(record)
    assert brief == to_markdown(record, mode="prose")
    assert len(brief) < len(to_markdown(record, mode="full")) / 2
    # The corroboration is what got dropped, not the idea or its refutation.
    assert "Kills it" in brief
    assert "Settles it" in brief
    assert "Source sentences" not in brief
    # ...and a reader is told detail was withheld, rather than left to assume
    # the short report is the whole record.
    assert "Not shown" in brief
    assert "--report-mode full" in brief


def test_the_brief_report_keeps_every_warning_the_full_one_has(
    graph: KnowledgeGraph,
) -> None:
    """Brief is a shorter view, not a softer one."""
    record = bundle(Generator(graph=graph, params=_params(), judge=FakeJudge()).run())
    brief = to_markdown(record)
    # Truncated coverage: the absence-of-evidence warning is not optional.
    assert "not** evidence of absence" in brief
    # The caveats themselves survive, not merely the word: a caveat shared by
    # every hypothesis is hoisted to the header rather than dropped.
    caveats = {c for h in record.hypotheses for c in h.caveats}
    assert caveats, "fixture must produce caveats for this to test anything"
    for caveat in caveats:
        opening = caveat.split(":")[0][:40]
        assert opening in brief, caveat


def test_a_caveat_every_hypothesis_shares_is_stated_once(
    graph: KnowledgeGraph,
) -> None:
    """Repetition is the enemy of a short report, but dropping is the enemy of
    a safe one -- so a shared caveat moves up, it does not disappear."""
    record = bundle(Generator(graph=graph, params=_params(), judge=FakeJudge()).run())
    assert len(record.hypotheses) > 1
    shared = frozenset.intersection(*(frozenset(h.caveats) for h in record.hypotheses))
    assert shared, "fixture must have a caveat common to every hypothesis"

    brief = to_markdown(record)
    for caveat in shared:
        opening = caveat.split(":")[0][:40]
        assert brief.count(opening) == 1, f"stated {brief.count(opening)}x: {caveat}"
    assert "Applies to every hypothesis below" in brief


def test_clipping_cuts_on_sentences_and_always_marks_the_cut() -> None:
    """An argument that ends mid-case must not look like one that ended.

    Clipping is the one place this module trades completeness for brevity, so
    it may only cut at a sentence boundary and must leave a visible mark.
    """
    from adapters.report.render import _clip

    short = "One sentence that fits."
    assert _clip(short, 100) == short  # nothing to do, nothing marked

    two = "First sentence here. Second sentence that pushes past the budget."
    clipped = _clip(two, 30)
    assert clipped == "First sentence here. …"
    assert "Second" not in clipped

    # A decimal is not a sentence boundary -- support 0.505 must survive whole.
    decimals = "Support fell to 0.505 in the recomputation. Then more text follows."
    assert _clip(decimals, 45).startswith("Support fell to 0.505 in the recomputation.")

    # A single sentence longer than the budget still gets cut and marked, on a
    # word boundary rather than mid-token.
    one_long = "A single very long sentence that simply will not fit anywhere"
    cut = _clip(one_long, 20)
    assert cut.endswith("…") and len(cut) <= 24 and not cut.startswith("A single very long s ")


def test_every_mode_keeps_the_signals_a_reader_must_not_miss(
    graph: KnowledgeGraph,
) -> None:
    """A mode changes the form, never the safety.

    This is the test that stops a new view from quietly becoming a softer one:
    whatever shape it renders in, it carries the absence-of-evidence warning
    and it names a rejected hypothesis as rejected.
    """
    judge = FakeJudge(cite="L-does-not-exist")  # every hypothesis fails citations
    record = bundle(Generator(graph=graph, params=_params(), judge=judge).run())
    assert any(
        i.code == "illegal_citation" for h in record.hypotheses for i in h.issues
    ), "fixture must actually produce a rejected hypothesis"

    for mode in report.MODE_NAMES:
        rendered = to_markdown(record, mode=mode)
        assert "not** evidence of absence" in rendered, mode
        assert "CITATION REJECTED" in rendered, mode


def test_every_mode_renders_every_hypothesis(graph: KnowledgeGraph) -> None:
    """A view that silently drops rows is worse than no view."""
    record = bundle(Generator(graph=graph, params=_params(), judge=FakeJudge()).run())
    assert len(record.hypotheses) > 1
    for mode in report.MODE_NAMES:
        rendered = to_markdown(record, mode=mode)
        for hypothesis in record.hypotheses:
            assert hypothesis.subject_name in rendered, (mode, hypothesis.id)


def test_trace_mode_names_every_link_and_its_evidence(graph: KnowledgeGraph) -> None:
    """Trace answers 'where did this come from', so the ids have to be in it."""
    record = bundle(Generator(graph=graph, params=_params(), judge=FakeJudge()).run())
    trace = to_markdown(record, mode="trace")
    for hypothesis in record.hypotheses:
        for step in hypothesis.path:
            assert step["link"] in trace
        for finding_id, finding in hypothesis.evidence["findings"].items():
            assert finding_id in trace
            # The verbatim sentence, not a paraphrase of it.
            assert finding["quote"] in trace


def test_table_mode_is_one_row_per_hypothesis(graph: KnowledgeGraph) -> None:
    record = bundle(Generator(graph=graph, params=_params(), judge=FakeJudge()).run())
    table = to_markdown(record, mode="table")
    # The `|---|` separator does not match this filter, so it is the header row
    # plus exactly one row per hypothesis.
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    assert len(rows) == len(record.hypotheses) + 1


def test_an_unknown_mode_is_an_error(graph: KnowledgeGraph) -> None:
    """Silently falling back to prose would hand an auditor a partial record
    that looks complete."""
    record = bundle(Generator(graph=graph, params=_params(), judge=FakeJudge()).run())
    with pytest.raises(ValueError, match="mode must be"):
        to_markdown(record, mode="verbose")


def test_report_names_the_failure_it_found(graph: KnowledgeGraph) -> None:
    """"Blocked before we spent a call" and "the model cited what it was never
    shown" are different diagnoses, and the badge has to say which."""
    from adapters.report.render import _failure_badges
    from hyp_gen.hypothesis import Hypothesis, ValidationIssue

    def badge(*codes: str) -> str:
        h = Hypothesis(
            id="h", motif="m", subject="a", object="b",
            subject_name="a", object_name="b", hops=1,
            issues=[ValidationIssue(code=c, detail="", severity="error") for c in codes],
        )
        return " ".join(_failure_badges(h))

    assert "BLOCKED" in badge("already_stated")
    assert "CITATION REJECTED" in badge("illegal_citation")
    both = badge("broken_path", "illegal_citation")
    assert "BLOCKED" in both and "CITATION REJECTED" in both
    assert badge() == ""

    # And a run whose model cites out of pack is labelled that way end to end.
    record = bundle(Generator(graph=graph, params=_params(), judge=FakeJudge(cite="L-nope")).run())
    assert "CITATION REJECTED" in to_markdown(record)


def test_the_report_shows_the_gate_table(graph: KnowledgeGraph) -> None:
    record = bundle(Generator(graph=graph, params=_params(), judge=FakeJudge()).run())
    markdown = to_markdown(record, mode="full")
    assert "**Verification**" in markdown
    assert "gate 1 structure" in markdown
    assert "VERDICT" in markdown


def test_a_halt_is_stated_in_prose_as_well_as_the_table(
    graph: KnowledgeGraph,
) -> None:
    """A halt is the one thing in the report a reader must not be able to
    mistake for a clean run."""
    strict = Params(
        selection=SelectionParams(top_k=2),
        evidence=EvidenceParams(min_independent_groups=99),
    )
    markdown = to_markdown(bundle(Generator(graph=graph, params=strict).run()))
    assert "Verification stopped at **independence**" in markdown
    assert "none of them should be read as passed" in markdown
