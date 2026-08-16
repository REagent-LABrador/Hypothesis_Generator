# Hypothesis Generation

A knowledge graph goes in. A set of hypotheses comes out, each one traceable to
a link id, a finding id, and the verbatim sentence a human wrote.

```
examples/knowledge-graph.json  ──▶  hypotheses.json
```

That is the whole application. Both sides are schemas this repo owns and
publishes:


|            | File                 | Schema                                                                       |
| ---------- | -------------------- | ---------------------------------------------------------------------------- |
| **Input**  | any graph you supply | `[schemas/knowledge-graph.schema.json](schemas/knowledge-graph.schema.json)` |
| **Output** | `hypotheses.json`    | `[schemas/hypotheses.schema.json](schemas/hypotheses.schema.json)`           |


Both schemas are generated from the pydantic models that read and write them
(`tools/generate_schemas.py`), and a test fails if the committed copies drift.
The contract cannot rot away from the code.

The app is deliberately blind to where the graph came from — no search
strategy, no PubMed, no query log. If a fact is not in the graph, it cannot
appear in a hypothesis. That is what makes the output checkable: every claim
resolves to a row in the input.

A run is a function of `(graph, params)` and nothing else. Same inputs, same
output, so a disagreement about the result is a disagreement about parameters
rather than about luck.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# Deterministic: enumeration, scoring, selection. No model calls, no API key.
.venv/bin/hypgen --graph examples/knowledge-graph.json --dry-run

# Full run — articulate, critique, check citations. Needs ANTHROPIC_API_KEY.
.venv/bin/hypgen --graph examples/knowledge-graph.json \
  --profile repurposing --out runs/first-run

.venv/bin/python -m pytest
```

`python -m hyp_gen` is the same entry point if you would rather not rely on the
console script.

Start with `--dry-run`. Most early failures are traversal or parameter
failures, and they are far easier to see as a table of candidates than inside a
finished report.

```
id                          motif                    sup   nov  test  risk   str    rank
H-analog-t1-t11-via-t2      analogical_transfer     0.71  0.34  0.70  0.00  1.00   0.461
    pirfenidone → systemic sclerosis ILD
    · g2 was searched in round 2
    ! independence: all primary evidence here is from Distler; nothing replicates it
H-g1                        gap_closure             0.46  0.39  0.55  0.33  0.54   0.454
    metformin → AMPK → collagen I deposition → idiopathic pulmonary fibrosis
```

The `!` and `✗` lines are deterministic verification gates, which need no API
key. `✗` is a candidate that will be thrown out — knowing that costs nothing
here and a model call later.

## `hypotheses.json` is the canonical artifact

Everything else this app can write is a **pure function of that one file**. No
view or export opens the graph or calls a model, which is why any of them can
be regenerated from a saved run — and why none of them can state anything the
record does not carry.

```
                        hypotheses.json   ← the record
                              │
        ┌─────────────────────┼──────────────────┬────────────────────┐
   report.md               traces.svg        webui.json        *.program.json
   --report-mode           --emit-diagram    --emit-webui      --emit-programs
   prose|table|trace|full
```

```bash
# Re-render anything from a saved run. Costs no model calls.
hypgen --report-from runs/first-run/hypotheses.json --report-mode table --out runs/first-run
hypgen --report-from runs/first-run/hypotheses.json --emit-diagram runs/first-run/traces.svg
```

A view changes the form, never the safety: failure badges, halted
verifications, error-level validation issues and the absence-of-evidence notice
render in all of them.

## Layout

```
src/hyp_gen/
  graph.py       INPUT contract — the knowledge graph this app accepts
  record.py      OUTPUT contract — HypothesisSet, written as hypotheses.json
  params.py      the knobs: profiles, the craziness dial, thresholds
  pipeline.py    the run: graph in, record out
  cli.py         the terminal surface

  generate/      deterministic pattern finding. No API key — this is --dry-run
  reasoning/     the only place a model is called
  checks/        what the model said, evaluated back against the graph
  views/         report · diagram · webui   derived from the record
  export/        valuation                  derived from the record

schemas/         the two contracts above, as JSON Schema (generated, committed)
tools/           generate_schemas.py
examples/        a knowledge graph and an analyst frame to run against
tests/           261 tests, offline, no network
```

The two contract files sit at the top level on purpose: they are the app's
boundary, and everything under them is how you get from one to the other. The
subpackages are grouped by what they need and what they trust — `generate/`
needs only the graph, `reasoning/` is the only thing that costs money,
`checks/` trusts neither the model nor the graph's own stated confidence, and
`views/`/`export/` may read nothing but the record.

## The pipeline

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
   └─ record.py                hypotheses.json
```

Model calls happen only for candidates that survive selection, so cost scales
with `selection.top_k`, not with graph size. Every model call is bracketed by
`checks/`: nothing `reasoning/` produces reaches the record without being
validated against the graph and the evidence pack it was shown.

## Where hypotheses come from

Four motifs, each a distinct reason a statement is worth making:


| Motif                 | The shape                                                    | The claim                                             |
| --------------------- | ------------------------------------------------------------ | ----------------------------------------------------- |
| `gap_closure`         | the graph flags a pair its own links imply but nobody states | the implied relation is real                          |
| `transitive_chain`    | A→B→C exists, A→C does not                                   | the chain composes                                    |
| `analogical_transfer` | X and Y share neighbours; X has an edge Y lacks              | Y has it too                                          |
| `condition_split`     | a link disagrees, under different `where` conditions         | both results are right; the condition is the variable |


`condition_split` is the one people are surprised by. A `disagreed` link is
usually two experimental conditions rather than a conflict, so reconciling it
is treated as a first-class hypothesis instead of a data quality problem.

## The parts that carry the design

- **Absence is not evidence of absence.** Novelty that rests on a gap is scaled
by the graph's own `absence_reliability()`, computed from coverage depth and
truncation. At `quick` depth that factor is zero: page one lies, so nothing
may claim to be new merely because this search did not surface it.
- **Support is recomputed, not trusted.** The input's `links.confidence` is
treated as a claim, not a fact. Support is recomputed from `findings` +
`papers`, with study type, hedging, secondhand citation, preprint status and
independent-group counts applied. `drift` reports where we differ.
- **Support and novelty are separate axes.** A fully supported hypothesis is a
known fact. Averaging the two ranks textbook statements first, so the scores
stay a vector and the Pareto front is in the record for whoever consumes it.
- **A chain is as strong as its weakest link.** Weakest-link aggregation is the
default because `mean` lets one strong link launder two weak ones.
- **Hubs are damped, not banned.** Degree-weighted path counts (Rephetio's
DWPC) stop "aspirin → inflammation → everything" from topping every run.
- **The model may only cite what it was shown.** Each candidate gets an
evidence pack, and any id outside it is rejected by `checks/validate.py` and
the hypothesis flagged. A model that cites `L7` when `L7` was never in its
pack has stopped reporting and started remembering.
- **Verification is a process with an order, and a skip is not a pass.** Six
gates per hypothesis, cheapest first, so the four deterministic ones can
reject a candidate before the adversarial gate spends a call. When one halts,
the rest are recorded as skipped *naming the halt* — five green checks
because the sixth never ran would read as more verified than the truth.
- **Critics get lenses, not copies.** Three identical refuters mostly agree; a
mechanism critic and an evidence critic fail on different things.
- **The loop closes by id.** Each hypothesis names the exact `resolve_link`,
`test_gap`, or `expand_node` request that would move it — no prose for the
graph builder to interpret.

## Profiles and the craziness dial

One graph, five stances. `--profile` picks one; `--set group.key=value` patches
any field on top.


| Profile        | For                                                                         |
| -------------- | --------------------------------------------------------------------------- |
| `default`      | balanced                                                                    |
| `conservative` | short paths, strong links, two independent groups, no reversals             |
| `speculative`  | longer paths, weaker links, more critics                                    |
| `repurposing`  | compound → gene/protein → process → disease                                 |
| `mechanism`    | closed discovery: both ends given, find the B terms                         |
| `valuation`    | shaped for the export: intervention in, disease out, ≤2 labels per molecule |


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
`rejected`), recorded per hypothesis in `hypotheses.json`:

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

## Status

**Working.** Graph parsing, typed/degree-weighted traversal, all four motifs,
evidence recomputation, multi-objective scoring, MMR selection with quotas,
evidence packs, staged six-gate verification with halting, articulation,
multi-lens critique, Elo tournament, evolution rounds, graph-builder asks, four
report modes, SVG traces, the web UI payload, the valuation export, CLI.

**Untested against the live API.** The model stages are exercised end-to-end by
a scripted fake judge (including refusal, budget exhaustion, and
illegal-citation paths), and the call shape is checked against `anthropic`
0.122.0, but the first real run should be a `--profile conservative` one-off
with `selection.top_k=2`.

**Not built.** Retrospective validation (hold out a round, check whether the
generator proposes what the later round found); multi-round driving of the
graph builder from `asks`; dataset-support scoring.