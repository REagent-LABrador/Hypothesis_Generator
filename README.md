# hyp_gen

A knowledge graph goes in. **One hypothesis** comes out, traceable to a link
id, a finding id, and the verbatim sentence a human wrote.

```
knowledge-graph.json  ──▶  [ hyp_gen ]  ──▶  hypothesis.json
```

That is the whole core. Both sides are contracts this repo owns, documented
beside the code that implements them:

| | Contract | Prose | Machine-readable |
|---|---|---|---|
| **Input** | the knowledge graph accepted | [`src/hyp_gen/INPUT_SCHEMA.md`](src/hyp_gen/INPUT_SCHEMA.md) | [`schemas/knowledge-graph.schema.json`](schemas/knowledge-graph.schema.json) |
| **Output** | one `HypothesisDocument` | [`src/hyp_gen/OUTPUT_SCHEMA.md`](src/hyp_gen/OUTPUT_SCHEMA.md) | [`schemas/hypothesis.schema.json`](schemas/hypothesis.schema.json) |

Every schema document is generated from the pydantic models that actually read
and write those files, and a test fails if a committed copy drifts. The
contract cannot rot away from the code.

## One hypothesis, not a slate

A run enumerates many candidates, scores them, critiques them and ranks them —
that is how it knows which one is best — and then writes **the winner, alone**.
If nothing survives selection it writes nothing and says so: an empty answer
with a clear next step is a real answer, and a better one than the least bad
candidate promoted to look like a finding.

The document is self-contained. `provenance` ships with the claim because a
hypothesis separated from what produced it cannot be judged: support 0.5 from a
cautious run and support 0.5 from an ambitious one are different statements
about the world, and the number alone cannot tell them apart.

## Core and adapters

The core does the science. Everything that turns a hypothesis into something
*else* is an adapter — a separate program that reads `hypothesis.json` and
never touches the graph.

```
                                  ┌─▶ adapters/report/     report.md
knowledge-graph.json              │
        │                         ├─▶ adapters/webui/      cards.json + traces.svg
        ▼                         │
   [ hyp_gen ] ─▶ hypothesis.json ─┤
                                  └─▶ adapters/valuation/  *.program.json  (ROI model)
```

The dependency runs one way: adapters import `hyp_gen`; `hyp_gen` never imports
an adapter, and no adapter imports another. A test enforces it. Each adapter
carries its own `INPUT_SCHEMA.md` and `OUTPUT_SCHEMA.md`, and the rules they all
obey — read documents not the graph, never call a model, add no claim, preserve
every warning — are in [`adapters/common.py`](adapters/common.py).

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# The core. Deterministic, no API key: enumeration, scoring, selection.
hypgen --graph examples/knowledge-graph.json --dry-run --out runs/first

# The full run — articulate, critique, check citations. Needs ANTHROPIC_API_KEY.
hypgen --graph examples/knowledge-graph.json --profile repurposing --out runs/first

# Adapters, over the document the core wrote.
hypreport    runs/first/hypothesis.json --mode prose --mode trace --out runs/first
hypwebui     runs/first/hypothesis.json --cards runs/first/cards.json --svg runs/first/traces.svg
hypvaluation runs/first/hypothesis.json --frame frame.json --out runs/first/programs
```

Without `--out` the core writes the document to stdout and its diagnostics to
stderr, so an adapter can be piped straight onto it:

```bash
hypgen --graph examples/knowledge-graph.json --dry-run | hypreport - --mode table
```

Start with `--dry-run`. Most early failures are traversal or parameter
failures, and they are far easier to see as a table of candidates than inside a
finished hypothesis.

```
id                          motif                    sup   nov  test  risk   str    rank
H-analog-t1-t11-via-t2      analogical_transfer     0.71  0.34  0.70  0.00  1.00   0.461
    pirfenidone → systemic sclerosis ILD
    ! independence: all primary evidence here is from Distler; nothing replicates it
H-g1                        gap_closure             0.46  0.39  0.55  0.33  0.54   0.454
    metformin → AMPK → collagen I deposition → idiopathic pulmonary fibrosis
```

The `!` and `✗` lines are deterministic verification gates, which need no API
key. `✗` is a candidate that will be thrown out — knowing that costs nothing
here and a model call later.

## Layout

```
src/hyp_gen/               THE CORE
  INPUT_SCHEMA.md          what a knowledge graph must look like
  OUTPUT_SCHEMA.md         what hypothesis.json is
  graph.py                 the input:  parse, index, typed degree-weighted traversal
  hypothesis.py            the output: HypothesisDocument
  params.py                the knobs: profiles, the craziness dial, thresholds
  pipeline.py              the run, and RunResult.top() — the boundary
  cli.py                   hypgen
  generate/                deterministic pattern finding. No API key — this is --dry-run
  reasoning/               the only place a model is called
  checks/                  what the model said, evaluated back against the graph

adapters/                  EVERYTHING ELSE
  common.py                the rules, the loader, the shared failure formatting
  report/                  + INPUT_SCHEMA.md  OUTPUT_SCHEMA.md
  webui/                   + INPUT_SCHEMA.md  OUTPUT_SCHEMA.md
  valuation/               + INPUT_SCHEMA.md  OUTPUT_SCHEMA.md

schemas/                   the core's two contracts as JSON Schema (generated)
tools/                     generate_schemas.py — regenerates every schema doc
examples/                  a knowledge graph and an analyst frame
tests/core/                the core flow
tests/adapters/            each adapter, and that it stays pure
```

## The core flow

```
knowledge-graph.json
   │
   ├─ graph.py                 parse + index + typed, degree-weighted traversal
   ├─ generate/candidates.py   four motifs → structural candidates
   ├─ generate/scoring.py      recompute support; novelty, risk, testability
   ├─ generate/select.py       thresholds → Pareto front → MMR → quotas
   ├─ generate/evidence.py     per-candidate pack: the model's entire world
   │                           ── everything above needs no API key ──
   ├─ reasoning/reason.py      articulate → critique from N lenses → compare → evolve
   ├─ checks/validate.py       structure against the graph, citations against the pack
   ├─ checks/verify.py         six gates in cost order; a halt skips the rest, loudly
   ├─ generate/asks.py         weakest point → one request back to the graph builder
   │
   └─ RunResult.top()          the winner, as hypothesis.json
```

Model calls happen only for candidates that survive selection, so cost scales
with `selection.top_k`, not with graph size. Every model call is bracketed by
`checks/`: nothing `reasoning/` produces reaches the document without being
validated against the graph and the evidence pack it was shown.

## Where hypotheses come from

Four motifs, each a distinct reason a statement is worth making:

| Motif | The shape | The claim |
|---|---|---|
| `gap_closure` | the graph flags a pair its own links imply but nobody states | the implied relation is real |
| `transitive_chain` | A→B→C exists, A→C does not | the chain composes |
| `analogical_transfer` | X and Y share neighbours; X has an edge Y lacks | Y has it too |
| `condition_split` | a link disagrees, under different `where` conditions | both results are right; the condition is the variable |

`condition_split` is the one people are surprised by. A `disagreed` link is
usually two experimental conditions rather than a conflict, so reconciling it
is treated as a first-class hypothesis instead of a data quality problem.

## The parts that carry the design

- **Absence is not evidence of absence.** Novelty that rests on a gap is scaled
  by the graph's own `absence_reliability()`, computed from coverage depth and
  truncation. At `quick` depth that factor is zero: page one lies, so nothing
  may claim to be new merely because this search did not surface it.
- **Support is recomputed, not trusted.** The input's `links.confidence` is a
  claim, not a fact. Support is recomputed from `findings` + `papers`, with
  study type, hedging, secondhand citation, preprint status and
  independent-group counts applied. `drift` reports where we differ.
- **Support and novelty are separate axes.** A fully supported hypothesis is a
  known fact. Averaging the two ranks textbook statements first, so the scores
  stay a vector.
- **A chain is as strong as its weakest link.** Weakest-link aggregation is the
  default because `mean` lets one strong link launder two weak ones.
- **Hubs are damped, not banned.** Degree-weighted path counts (Rephetio's
  DWPC) stop "aspirin → inflammation → everything" winning every run.
- **The model may only cite what it was shown.** Each candidate gets an
  evidence pack, and any id outside it is rejected by `checks/validate.py`. A
  model that cites `L7` when `L7` was never in its pack has stopped reporting
  and started remembering.
- **Verification is a process with an order, and a skip is not a pass.** Six
  gates, cheapest first, so the four deterministic ones can reject a candidate
  before the adversarial gate spends a call. When one halts, the rest are
  recorded as skipped *naming the halt* — five green checks because the sixth
  never ran would read as more verified than the truth.
- **Critics get lenses, not copies.** Three identical refuters mostly agree; a
  mechanism critic and an evidence critic fail on different things.
- **The loop closes by id.** The hypothesis names the exact `resolve_link`,
  `test_gap` or `expand_node` request that would move it — no prose to
  interpret.

## Profiles and the craziness dial

One graph, five stances. `--profile` picks one; `--set group.key=value` patches
any field on top.

| Profile | For |
|---|---|
| `default` | balanced |
| `conservative` | short paths, strong links, two independent groups, no reversals |
| `speculative` | longer paths, weaker links, more critics |
| `repurposing` | compound → gene/protein → process → disease |
| `mechanism` | closed discovery: both ends given, find the B terms |
| `valuation` | shaped for the valuation adapter: intervention in, disease out |

```bash
# Why might metformin act on IPF?  (closed discovery)
--profile mechanism --set framing.anchors='["metformin"]' --set framing.targets='["IPF"]'
```

`--craziness` is one float from 0 to 1. A profile picks *what question* to ask
the graph; craziness picks *how far out* to reach for an answer. At 0.0 you get
two-hop chains between strongly-supported links corroborated by two independent
groups — nearly boring, which is the point when the next step costs money. At
1.0 the similarity motif leads and cross-kind analogy is allowed.

The dial widens the aperture. It never lowers the audit standard: the same
chain scores the same support at either end, absence still is not evidence of
absence, a hypothesis still may not cite what it was not shown, and the
`structure` and `citations` gates still halt. Scrutiny goes *up* with ambition —
1.0 buys a third critic and a revision round, because that end's failure mode
is fluent nonsense.

## Verification

Six gates, one of four verdicts (`verified` / `qualified` / `unverified` /
`rejected`), recorded in the document:

```
gate 1 structure       PASS   3 hop(s), path intact, not already stated
gate 2 citations       PASS   4 ids, all legal
gate 3 consistency     PASS   5 claims, 3 grounded in evidence
gate 4 independence    FAIL   F2, F7 share first author Distler — 1 group, run requires 2
gate 5 falsifiability  SKIP   halted at independence
gate 6 adversarial     SKIP   halted at independence
──────────────────────────────────────────────────────────────────────────
VERDICT  unverified (halted: independence)
```

## Development

```bash
.venv/bin/python -m pytest                    # 265 tests, offline, no network
.venv/bin/python tools/generate_schemas.py    # after changing any contract model
```

Changing a field on `KnowledgeGraph`, `HypothesisDocument`, `WebPayload`,
`Emission` or `ProgramFrame` means regenerating the schema documents;
`tests/core/test_contracts.py` fails until you do.

## Status

**Working.** Graph parsing, typed/degree-weighted traversal, all four motifs,
evidence recomputation, multi-objective scoring, MMR selection with quotas,
evidence packs, staged six-gate verification with halting, articulation,
multi-lens critique, Elo tournament, evolution rounds, graph-builder asks, and
all three adapters.

**Untested against the live API.** The model stages are exercised end to end by
a scripted fake judge (including refusal, budget exhaustion and
illegal-citation paths), and the call shape is checked against `anthropic`
0.122.0, but the first real run should be a `--profile conservative` one-off
with `selection.top_k=2`.

**Not built.** Retrospective validation (hold out a round, check whether the
generator proposes what the later round found); multi-round driving of the
graph builder from `asks`; dataset-support scoring.
