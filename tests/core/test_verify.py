"""The staged verification process: order, halting, and what a skip means.

The tests that matter most here are not the ones checking that a bad hypothesis
fails. They are the ones checking that when it fails, everything downstream is
recorded as *not run* rather than as passed, and that the failure does not
quietly delete the hypothesis from the record. A verification process whose
failures are invisible is worse than none, because it reads as assurance.
"""

from __future__ import annotations

from dataclasses import replace

from hyp_gen.generate.candidates import Candidate, enumerate_candidates
from hyp_gen.generate.evidence import build_pack
from hyp_gen.graph import Edge, GraphIndex
from hyp_gen.params import EvidenceParams, Params, VerificationParams
from hyp_gen.hypothesis import Articulation, Claim, Critique, Hypothesis
from hyp_gen.generate.scoring import score_candidate
from hyp_gen.checks.verify import GateContext, apply, verify


def _with_path(candidate: Candidate, *edges: Edge) -> Candidate:
    """Both ``Candidate`` and ``Edge`` are frozen, which is what keeps a graph
    fixture from leaking a mutation into the next test. Damaging a path for a
    test therefore means rebuilding it, not poking it."""
    return replace(candidate, path=tuple(edges))


def _context(index: GraphIndex, params: Params, motif: str, **kwargs) -> GateContext:
    candidate = next(c for c in enumerate_candidates(index, params) if c.motif == motif)
    pack = build_pack(index, candidate, score_candidate(index, candidate, params))
    return GateContext(
        index=index, candidate=candidate, pack=pack, params=params, **kwargs
    )


def _articulation(ctx: GateContext, **overrides) -> Articulation:
    """A well-formed articulation citing this candidate's own links.

    Built from the pack rather than hard-coded so the citation gate sees legal
    ids and the tests below isolate the gate they are actually about.
    """
    cites = list(ctx.candidate.link_ids)[:1] or list(ctx.pack.findings)[:1]
    fields = {
        "statement": "Pirfenidone reduces fibrotic remodelling in this tissue.",
        "mechanism": "A -> X -> B",
        "claims": [Claim(text="the compound engages the target", cites=cites)],
        "novel_because": "the graph never states this pair directly",
        "predictions": ["target engagement precedes the phenotype"],
        "falsifier": "the phenotype persists with the target genetically deleted",
        "decisive_experiment": "knock out the target and measure the phenotype",
        "assumptions": [],
    }
    fields.update(overrides)
    return Articulation(**fields)


# -- the process ----------------------------------------------------------


def test_gates_run_in_the_declared_order(index: GraphIndex, params: Params) -> None:
    ctx = _context(index, params, "transitive_chain")
    names = [g.name for g in verify(ctx).gates]
    assert names == list(params.verification.gates)


def test_deterministic_gates_run_without_a_judge(
    index: GraphIndex, params: Params
) -> None:
    """--dry-run and a keyless demo both depend on this."""
    ctx = _context(index, params, "transitive_chain")
    result = verify(ctx)
    assert result.gate("structure").status == "pass"
    assert result.gate("independence").status in ("pass", "warn")
    assert result.gate("adversarial").status == "skip"


def test_a_halt_marks_everything_downstream_skipped(
    index: GraphIndex, params: Params
) -> None:
    """The failure mode this whole module exists to prevent: five green checks
    because the sixth never ran."""
    ctx = _context(index, params, "transitive_chain")
    ctx.candidate = _with_path(
        ctx.candidate, ctx.candidate.path[0].model_copy(update={"src": "t-nonexistent"})
    )

    result = verify(ctx)
    assert result.halted_at == "structure"
    assert result.verdict == "rejected"

    downstream = [g for g in result.gates if g.name != "structure"]
    assert all(g.status == "skip" for g in downstream)
    assert all("halted at structure" in g.summary for g in downstream)


def test_a_skip_is_never_a_pass(index: GraphIndex, params: Params) -> None:
    """A run that could not check something must not read as verified."""
    ctx = _context(index, params, "transitive_chain")  # no articulation, no judge
    result = verify(ctx)
    assert any(g.status == "skip" for g in result.gates)
    assert result.verdict != "verified"


def test_unknown_gate_names_do_not_crash_a_finished_run(
    index: GraphIndex, params: Params
) -> None:
    params = params.model_copy(
        update={"verification": VerificationParams(gates=("structure", "nonsense"))}
    )
    result = verify(_context(index, params, "transitive_chain"))
    assert result.gate("nonsense").status == "skip"
    assert result.gate("nonsense").summary == "unknown gate name"


def test_dropping_a_gate_removes_it_rather_than_skipping_it(
    index: GraphIndex, params: Params
) -> None:
    """"We chose not to check" and "we could not check" are different things,
    and the table must not blur them."""
    params = params.model_copy(
        update={"verification": VerificationParams(gates=("structure",))}
    )
    result = verify(_context(index, params, "transitive_chain"))
    assert [g.name for g in result.gates] == ["structure"]


# -- consistency ----------------------------------------------------------


def test_an_articulation_citing_no_evidence_fails_consistency(
    index: GraphIndex, params: Params
) -> None:
    """Legally cited and still ungrounded: every citation points at a node, so
    the citation gate is satisfied and nothing rests on evidence."""
    ctx = _context(index, params, "transitive_chain")
    node = next(iter(ctx.pack.things))
    ctx.articulation = _articulation(ctx, claims=[Claim(text="a claim", cites=[node])])

    gate = verify(ctx).gate("consistency")
    assert gate.status == "fail"
    assert {i.code for i in gate.issues} == {"ungrounded"}


def test_all_inferred_claims_fail_consistency(
    index: GraphIndex, params: Params
) -> None:
    ctx = _context(index, params, "transitive_chain")
    link = ctx.candidate.link_ids[0]
    ctx.articulation = _articulation(
        ctx, claims=[Claim(text="a reasoned step", cites=[link], inferred=True)]
    )

    gate = verify(ctx).gate("consistency")
    assert gate.status == "fail"
    assert "all_inferred" in {i.code for i in gate.issues}


def test_duplicate_claims_are_warned_not_failed(
    index: GraphIndex, params: Params
) -> None:
    ctx = _context(index, params, "transitive_chain")
    link = ctx.candidate.link_ids[0]
    ctx.articulation = _articulation(
        ctx,
        claims=[
            Claim(text="the compound engages the target", cites=[link]),
            Claim(text="The compound engages the target.", cites=[link]),
        ],
    )

    gate = verify(ctx).gate("consistency")
    assert gate.status == "warn"
    assert "duplicate_claim" in {i.code for i in gate.issues}


# -- independence ---------------------------------------------------------


def test_single_group_support_warns_by_default(
    index: GraphIndex, params: Params
) -> None:
    """The default profile asks for one group, so one group is a warning about
    replication rather than a failure."""
    ctx = _context(index, params, "analogical_transfer")
    gate = verify(ctx).gate("independence")
    assert gate.status == "warn"
    assert "single_group" in {i.code for i in gate.issues}


def test_conservative_params_turn_single_group_into_a_halt(
    index: GraphIndex, params: Params
) -> None:
    """Same evidence, stricter run: `evidence.min_independent_groups` is what
    decides, so strictness is a parameter you can diff between profiles."""
    strict = params.model_copy(
        update={"evidence": EvidenceParams(min_independent_groups=2)}
    )
    ctx = _context(index, strict, "analogical_transfer")
    result = verify(ctx)

    assert result.gate("independence").status == "fail"
    assert result.halted_at == "independence"
    assert result.verdict == "unverified"
    assert result.gate("adversarial").status == "skip"


def test_independence_counts_only_primary_results(
    index: GraphIndex, params: Params
) -> None:
    """A paper citing someone else's result is not a second group."""
    ctx = _context(index, params, "transitive_chain")
    for finding in ctx.pack.findings.values():
        finding["is_own_result"] = False

    gate = verify(ctx).gate("independence")
    assert gate.status == "fail"
    assert "no_primary_evidence" in {i.code for i in gate.issues}


# -- falsifiability -------------------------------------------------------


def test_a_falsifier_that_restates_the_hypothesis_fails(
    index: GraphIndex, params: Params
) -> None:
    ctx = _context(index, params, "transitive_chain")
    statement = "Pirfenidone reduces fibrotic remodelling in this tissue."
    ctx.articulation = _articulation(
        ctx, statement=statement, falsifier=f"{statement} is not true"
    )

    gate = verify(ctx).gate("falsifiability")
    assert gate.status == "fail"
    assert "vacuous_falsifier" in {i.code for i in gate.issues}


def test_a_real_falsifier_passes(index: GraphIndex, params: Params) -> None:
    ctx = _context(index, params, "transitive_chain")
    ctx.articulation = _articulation(ctx)
    assert verify(ctx).gate("falsifiability").status == "pass"


def test_falsifiability_runs_before_the_model_gate(
    index: GraphIndex, params: Params
) -> None:
    """Order is cost: a hypothesis nothing could falsify is not worth critics."""
    gates = list(params.verification.gates)
    assert gates.index("falsifiability") < gates.index("adversarial")


# -- adversarial ----------------------------------------------------------


def test_critics_that_refute_fail_the_gate(index: GraphIndex, params: Params) -> None:
    ctx = _context(index, params, "transitive_chain")
    ctx.articulation = _articulation(ctx)
    ctx.critics = lambda: [
        Critique(verdict="unsupported", strongest_objection="nothing backs the middle link", lens=lens)
        for lens in ("mechanism", "evidence")
    ]

    result = verify(ctx)
    assert result.gate("adversarial").status == "fail"
    assert ctx.verdict == "unsupported"
    assert result.verdict == "unverified"


def test_critics_citing_evidence_they_were_not_shown_fail_the_gate(
    index: GraphIndex, params: Params
) -> None:
    """A critic that invents an id is as untrustworthy as an articulator that
    does, and this is the pass that catches it."""
    ctx = _context(index, params, "transitive_chain")
    ctx.articulation = _articulation(ctx)
    ctx.critics = lambda: [
        Critique(
            verdict="supported",
            strongest_objection="none",
            per_claim=[
                {"claim_index": 0, "verdict": "supported", "reason": "ok", "cites": ["L-imaginary"]}
            ],
            lens="evidence",
        )
    ]

    gate = verify(ctx).gate("adversarial")
    assert gate.status == "fail"
    assert "illegal_citation" in {i.code for i in gate.issues}


def test_no_critics_is_a_skip_not_an_approval(
    index: GraphIndex, params: Params
) -> None:
    """A critic that never ran did not approve anything."""
    ctx = _context(index, params, "transitive_chain")
    ctx.articulation = _articulation(ctx)
    ctx.critics = list  # runs, produces nothing (budget exhausted, refusal)

    result = verify(ctx)
    assert result.gate("adversarial").status == "skip"
    assert result.verdict != "verified"


# -- applying to a hypothesis ---------------------------------------------


def test_gate_failures_do_not_block_the_hypothesis(
    index: GraphIndex, params: Params
) -> None:
    """`blocked` keys on error-severity issues and controls whether a
    hypothesis reaches the record at all. The gates added here express
    themselves through the verdict instead: a hypothesis that fails
    independence must still be published, with the failure visible."""
    strict = params.model_copy(
        update={"evidence": EvidenceParams(min_independent_groups=2)}
    )
    ctx = _context(index, strict, "analogical_transfer")
    hypothesis = Hypothesis(
        id=ctx.candidate.id,
        motif=ctx.candidate.motif,
        subject=ctx.candidate.subject,
        object=ctx.candidate.object,
        subject_name="x",
        object_name="y",
        hops=ctx.candidate.hops,
    )

    apply(hypothesis, ctx)
    assert hypothesis.verification.verdict == "unverified"
    assert not hypothesis.blocked


def test_apply_does_not_duplicate_issues_found_at_assembly(
    index: GraphIndex, params: Params
) -> None:
    """`check_structure` already ran during assembly, so re-running it inside
    the process must not double every structural warning in the report."""
    ctx = _context(index, params, "gap_closure")
    hypothesis = Hypothesis(
        id=ctx.candidate.id,
        motif=ctx.candidate.motif,
        subject=ctx.candidate.subject,
        object=ctx.candidate.object,
        subject_name="x",
        object_name="y",
        hops=ctx.candidate.hops,
        issues=list(ctx.pack and []),
    )
    apply(hypothesis, ctx)
    first = len(hypothesis.issues)
    apply(hypothesis, ctx)
    assert len(hypothesis.issues) == first


def test_the_table_names_the_halting_gate(index: GraphIndex, params: Params) -> None:
    strict = params.model_copy(
        update={"evidence": EvidenceParams(min_independent_groups=2)}
    )
    ctx = _context(index, strict, "analogical_transfer")
    table = verify(ctx).table()

    assert "gate 1 structure" in table
    assert "SKIP" in table
    assert "VERDICT  unverified (halted: independence)" in table


# -- the bug the process surfaced -----------------------------------------


def test_analogical_transfer_is_not_blocked_by_its_own_bridge(
    index: GraphIndex, params: Params
) -> None:
    """Regression. An analogical candidate's path is the *donor's* bridge edge,
    because the proposal is precisely that the receiver lacks that link. A
    structure check that walks the path from `subject` therefore reported
    `broken_path` on every analogical hypothesis ever generated, and
    `broken_path` is an error, so all of them were blocked before articulation
    -- including the top-ranked row in the README's own example output.
    """
    ctx = _context(index, params, "analogical_transfer")
    issues = verify(ctx).gate("structure").issues
    assert [i for i in issues if i.code == "broken_path"] == []


def test_a_bridge_that_lands_somewhere_else_is_still_broken(
    index: GraphIndex, params: Params
) -> None:
    """Relaxing where the path may *start* must not relax where it ends."""
    ctx = _context(index, params, "analogical_transfer")
    ctx.candidate = _with_path(
        ctx.candidate, ctx.candidate.path[0].model_copy(update={"dst": "t-elsewhere"})
    )

    gate = verify(ctx).gate("structure")
    assert gate.status == "fail"
    assert "broken_path" in {i.code for i in gate.issues}
