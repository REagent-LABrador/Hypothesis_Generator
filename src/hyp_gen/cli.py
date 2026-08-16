"""The core command line: one graph in, one hypothesis out.

    hypgen --graph knowledge-graph.json --profile repurposing --out runs/first

Writes exactly one ``hypothesis.json`` -- the hypothesis that ranked first.
Reports, UI payloads, SVG traces and valuation briefs are not this program's
job; they are adapters that read the file this one writes. See ``adapters/``.

``--dry-run`` skips every model call and prints the structural shortlist: what
was enumerated, how it scored, and why. Start there. Most early failures are
traversal or parameter failures, and they are far easier to see as a table of
candidates than inside a finished hypothesis.
"""

from __future__ import annotations

import argparse
import builtins
import json
import sys
from pathlib import Path

from hyp_gen.checks import verify
from hyp_gen.generate.evidence import build_pack
from hyp_gen.graph import GraphIndex, KnowledgeGraph
from hyp_gen.params import PROFILES, Params
from hyp_gen.pipeline import Generator
from hyp_gen.reasoning.llm import Judge

FILENAME = "hypothesis.json"


def _overrides(pairs: list[str]) -> dict:
    """Parse ``--set traversal.max_hops=4`` into a nested dict.

    Values are parsed as JSON when possible so ``--set framing.mode=closed``
    and ``--set traversal.max_hops=4`` both do the obvious thing.
    """
    out: dict[str, dict] = {}
    for pair in pairs:
        if "=" not in pair or "." not in pair.split("=", 1)[0]:
            raise SystemExit(f"--set expects group.key=value, got {pair!r}")
        path, raw = pair.split("=", 1)
        group, key = path.split(".", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out.setdefault(group, {})[key] = value
    return out


def _dry_run(generator: Generator) -> None:
    """The shortlist, on stderr.

    Diagnostics go to stderr so stdout carries nothing but the document. That
    is what lets the core be piped straight into an adapter without a file in
    between, which is the point of adapters being separate programs.
    """

    def print(*args, **kwargs):  # noqa: A001 - deliberately shadowed
        builtins.print(*args, **kwargs, file=sys.stderr)

    index: GraphIndex = generator.index
    rows = generator.preview()

    print(f"graph {generator.graph.graph_id} round {generator.graph.round}")
    print(
        f"  {len(generator.graph.things)} things, {len(generator.graph.links)} links, "
        f"{len(generator.graph.findings)} findings, {len(generator.graph.gaps)} gaps"
    )
    print(
        f"  coverage: {generator.graph.coverage.depth}, "
        f"read {generator.graph.coverage.read}/{generator.graph.coverage.found}"
        f"{', truncated' if generator.graph.coverage.truncated else ''}"
        f"  →  absence_reliability {index.absence_reliability()}"
    )
    # The stance is the single most consequential choice a caller makes, and it
    # is invisible in the table below. Print what it actually resolved to, so a
    # surprising record is traceable to the dial rather than to the graph.
    stance, traversal = generator.params.stance, generator.params.traversal
    dial = "—" if stance.craziness is None else f"{stance.craziness:.2f}"
    print(
        f"  stance: profile {stance.profile}, craziness {dial}"
        f"  →  {traversal.max_hops} hops, links ≥ {traversal.min_link_confidence}, "
        f"{generator.params.evidence.min_independent_groups} independent group(s)"
    )
    print()
    header = f"{'id':<28}{'motif':<22}{'sup':>6}{'nov':>6}{'test':>6}{'risk':>6}{'str':>6}{'rank':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        candidate, scores = row.candidate, row.scores
        print(
            f"{candidate.id[:27]:<28}{candidate.motif:<22}"
            f"{scores.support:>6.2f}{scores.novelty:>6.2f}{scores.testability:>6.2f}"
            f"{scores.contradiction_risk:>6.2f}{scores.structure:>6.2f}"
            f"{scores.rank_score:>8.3f}"
        )
        print(f"    {' → '.join(row.chain)}")
        for note in scores.notes:
            print(f"    · {note}")

        # The deterministic gates need no key, so a dry run can already say
        # which candidates would be rejected on structure or on resting entirely
        # on one lab. Those are the two failures worth knowing about before
        # spending a single model call.
        for gate in row.warnings:
            mark = "✗" if gate.status == "fail" else "!"
            print(f"    {mark} {gate.name}: {gate.summary}")
    if not rows:
        print("(nothing survived selection — loosen the params or check the graph)")
    print(f"\n{len(rows)} shortlisted. No model calls made.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hypgen")
    parser.add_argument("--graph", type=Path, help="input knowledge graph JSON (see schemas/knowledge-graph.schema.json)")
    parser.add_argument("--profile", default="default", choices=sorted(PROFILES))
    parser.add_argument(
        "--focus-thing-id",
        help=(
            "only consider hypotheses whose evidence contains this graph thing id; "
            "one focused invocation still emits exactly one hypothesis"
        ),
    )
    parser.add_argument(
        "--craziness",
        type=float,
        metavar="0.0-1.0",
        help=(
            "how far out to reach: 0 is super safe (short paths, strong links, two "
            "independent groups), 1 is very ambitious (long paths, weak links, "
            "cross-kind analogy). Composes with --profile, which sets the question "
            "rather than the appetite."
        ),
    )
    parser.add_argument("--params", type=Path, help="params JSON, overrides --profile")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="GROUP.KEY=VALUE",
        help="patch one parameter, e.g. --set traversal.max_hops=4",
    )
    parser.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help=f"directory to write {FILENAME} into; prints to stdout without it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="no model calls: print the structural shortlist and stop short of reasoning",
    )
    args = parser.parse_args(argv)

    if args.graph is None:
        parser.error("--graph is required")

    graph = KnowledgeGraph.load(args.graph)
    if args.focus_thing_id and args.focus_thing_id not in {thing.id for thing in graph.things}:
        parser.error(
            f"--focus-thing-id {args.focus_thing_id!r} is not present in graph {graph.graph_id}"
        )
    overrides = _overrides(args.set)
    if args.params:
        if args.craziness is not None:
            parser.error("--params is a complete parameter set; --craziness derives one")
        base = Params.load(args.params).model_dump()
        for group, values in overrides.items():
            base.setdefault(group, {}).update(values)
        params = Params.model_validate(base)
    elif args.craziness is not None:
        try:
            params = Params.at_craziness(args.craziness, args.profile, overrides)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        params = Params.profile(args.profile, overrides)

    judge = None
    if not args.dry_run:
        try:
            judge = Judge(max_calls=params.budget.max_model_calls)
            ready = judge.has_credentials()
        except Exception as exc:  # pragma: no cover - defensive
            judge, ready, detail = None, False, str(exc)
        else:
            detail = "no API key, auth token, or profile credential resolved"
        if not ready:
            # A stack trace forty frames deep is a terrible first experience,
            # and the useful half of this pipeline needs no credentials at all.
            print(
                f"cannot reach the Anthropic API: {detail}.\n\n"
                "Set ANTHROPIC_API_KEY (or run `ant auth login`) for the full "
                "run, or use --dry-run for the structural record — it needs no "
                "credentials and still writes --out.",
                file=sys.stderr,
            )
            return 2

    generator = Generator(
        graph=graph,
        params=params,
        judge=judge,
        focus_thing_id=args.focus_thing_id,
    )

    if args.dry_run:
        _dry_run(generator)

    result = generator.run()
    document = result.top()

    if document is None:
        # An empty answer with a clear next step is a real answer, and it is a
        # better one than the least bad row promoted to look like a finding.
        print(
            "no hypothesis survived selection — nothing is written.\n"
            "Loosen the stance (--craziness, --profile speculative) or check the "
            "graph: at this stance it supports nothing worth stating.",
            file=sys.stderr,
        )
        return 1

    payload = document.model_dump_json(indent=2)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / FILENAME
        path.write_text(payload)
        print(
            f"wrote {path} — {document.hypothesis.id}, "
            f"chosen from {document.provenance.considered} considered",
            file=sys.stderr,
        )
        return 0

    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
