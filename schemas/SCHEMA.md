# Hypothesis Generation — Input ⇄ Output Contract

**This is the authoritative contract for the core.** Everything a consumer
needs is here: the graph you send, every field of the hypothesis that comes
back, the closed vocabularies, and the guarantees.
[`README.md`](../README.md) summarises this file for orientation; where the two
disagree, this file is correct.

```
knowledge-graph.json  ──▶  [ hyp_gen ]  ──▶  hypothesis.json
```

One graph in. **One hypothesis out** — the one that ranked first, not a slate
and not a list. Many candidates are enumerated, scored, critiqued and ranked on
the way there, because that is how the winner is known; only the winner crosses
this boundary.

| | file | schema | machine-readable |
|---|---|---|---|
| **Input** | any graph you supply | `KnowledgeGraph` | [`knowledge-graph.schema.json`](./knowledge-graph.schema.json) |
| **Output** | `hypothesis.json` | `HypothesisDocument` | [`hypothesis.schema.json`](./hypothesis.schema.json) |

The app is blind to where the graph came from — no search strategy, no PubMed,
no query log. **If a fact is not in the graph, it cannot appear in the
output.** That is the property that makes a hypothesis checkable: every claim
resolves to a row in the input.

A run is a function of `(graph, params)` and nothing else. Same inputs, same
output, so a disagreement about the result is a disagreement about parameters
rather than about luck.

---

## Input — the knowledge graph

One JSON file. Five lists referencing each other by `id`, plus a `coverage`
block saying how hard the graph was searched. Nothing nested.

| list | one row is | what this app does with it |
|---|---|---|
| `things` | a molecule, gene, protein, disease, process, method | the nodes it walks |
| `papers` | one source | study type, recency, and *who wrote it* — independence |
| `findings` | one claim from one paper + its exact sentence | support is recomputed from these |
| `links` | one relationship, summarising its findings | the edges it walks |
| `gaps` | a relationship implied but never stated | the `gap_closure` motif, and novelty |

`findings` is the raw evidence; `links` is the summary of it. Same data, two
levels — and this app reads both, because it does not take the summary's word
for it.

**Every field, and what it does:**

```jsonc
{
  "schema_version": "1.1",
  "graph_id": "g_demo1",        // copied into the output's provenance
  "question": "can an existing small molecule be repositioned against IPF?",
  "round": 2,                   // how many search passes built this graph
  "generated_at": "2026-08-15T09:12:00Z",

  "rounds": [                   // what each round asked. Read for `searched_in_round`
    { "n": 1, "ask": "new_question", "target": null,
      "depth": "standard", "papers_added": 25 },
    { "n": 2, "ask": "resolve_link", "target": "L2",
      "depth": "deep", "papers_added": 18 }
  ],

  // THE MOST IMPORTANT BLOCK IN THE FILE. Novelty that rests on a gap is
  // scaled by these numbers -- see note 1.
  "coverage": {
    "depth": "deep",            // quick | standard | deep | exhaustive
    "found": 318,               // results the search reported existing
    "read": 36,                 // ...of which this many were actually read
    "used": 33,                 // ...of which this many yielded a finding
    "truncated": true,          // true = a sample, not the literature
    "no_quote_discarded": 4,    // claims dropped for having no verbatim sentence
    "limits": {
      "max_papers": 50, "max_queries": 6,
      "hit_limit": "max_papers" // non-null = the budget stopped it, not the corpus
    }
  },

  "things": [{
    "id": "t1", "name": "pirfenidone",
    "kind": "small_molecule",   // protein|small_molecule|gene|disease|process|method
    "aliases": ["Esbriet"],     // surface forms merged into this node
    "mentions": 14              // papers mentioning it -- damps hubs, see note 5
  }],

  "papers": [{
    "id": "p9", "title": "Nintedanib slows decline in SSc-ILD",
    "year": 2019, "journal": "N Engl J Med", "doi": "10.1056/x9",
    "first_author": "Distler",  // THE independence key. Two findings sharing a
                                // first author are one research group, not two.
    "study_type": "clinical_trial",
    // meta_analysis|clinical_trial|human_cohort|animal|test_tube|computational|review
    "is_preprint": false,
    "round": 1
  }],

  "findings": [{
    "id": "f9", "from": "t2", "how": "slows", "to": "t11",
    "says": "yes",              // yes | no | no_effect -- `no` ARGUES AGAINST the link
    "quote": "nintedanib reduced the annual rate of FVC decline in SSc-ILD",
                                // the exact source words. No quote, no citation.
    "paper": "p9",
    "where": "phase 3 trial",   // the conditions measured under -- see note 4
    "is_own_result": true,      // false = citing someone else's work; excluded
                                // from independence, so a review cannot manufacture
                                // consensus
    "hedged": false,            // "may", "suggests", "could" -- discounts support
    "confidence": 0.95,         // the extractor's own read-accuracy score
    "flags": [],
    "round": 1
  }],

  "links": [{
    "id": "L8", "from": "t2", "how": "slows", "to": "t11",
    "yes": ["f9"], "no": [], "no_effect": [],   // findings on each side
    "state": "single_source",   // agreed|disagreed|single_source|no_effect
    "why": null,                // set when disagreed: usually conditions, not conflict
    "basis": "primary",         // primary|hedged_only|background_only|mixed
    "confidence": {             // TREATED AS A CLAIM, NOT A FACT -- see note 3
      "overall": 0.8, "label": "high",
      "evidence_quality": 0.9, "agreement": 1.0, "independence": 0.0
    },
    "changed_in_round": 2
  }],

  "gaps": [{
    "id": "g1", "missing": ["t1", "t3"],   // the pair nobody connected
    "implied_by": ["L1", "L2"],            // the links that suggest it
    "note": "nobody states metformin activates AMPK in lung fibroblasts",
    "confidence": 0.34,
    "searched_in_round": null   // null = NOBODY LOOKED. A searched-and-not-found
                                // gap is a far stronger novelty claim -- see note 2
  }]
}
```

**Only `graph_id` is strictly required.** Every list defaults to empty and every
scalar has a default, so a graph missing a block parses rather than refusing.
It will simply support fewer hypotheses, which is a truthful outcome.

**Unknown fields are preserved, not rejected.** The graph builder owns this
schema and will grow it; a new key upstream is not a reason for this app to stop
working. Nothing this app writes depends on a field it does not understand.

### The three fields this app leans on hardest

- **`coverage`** — depth, how much of what was found was read, and whether the
  search truncated. `absence_reliability` is computed from exactly these:
  `[quick 0.0, standard 0.45, deep 0.8, exhaustive 1.0]`, then ×0.6 if
  `truncated`, then ×0.85 if `limits.hit_limit` is set. At `quick` the factor is
  **zero** — page one lies, so nothing may claim to be new merely because this
  search did not surface it.
- **`findings`** — support is recomputed from these plus `papers`, never trusted
  from `links.confidence`. A link asserting 0.9 backed by one hedged preprint
  will not score like one backed by two independent groups.
- **`links.where`** (via its findings) — the conditions a result was measured
  under. A link whose findings disagree is usually two experimental conditions
  rather than a contradiction, which is why reconciling one is a first-class
  hypothesis here (`condition_split`) rather than a data-quality complaint.

---

## Output — one hypothesis

`hypothesis.json`, schema version **2.0**. Adding an optional field is a minor
bump; removing one, or narrowing what it may contain, is a major one.

If nothing survives selection, the run **writes nothing** and says so on stderr,
exiting 1. An empty answer with a clear next step is a real answer, and a better
one than the least bad candidate promoted to look like a finding.

**Every field, and what it means:**

```jsonc
{
  "schema_version": "2.0",

  // WHAT THE RUN WAS. Travels with the claim because a hypothesis separated
  // from what produced it cannot be judged -- see note 6.
  "provenance": {
    "graph_id": "g_demo1",      // the graph this read; copied from the input
    "round": 2,
    "question": "can an existing small molecule be repositioned against IPF?",
    "generated_at": "2026-08-15T09:12:00Z",
    "params": { /* the full resolved parameter set: stance, framing, traversal,
                   motifs, evidence, novelty, selection, ranking, loop, budget.
                   With the graph, these determine the output exactly. */ },
    "coverage": { /* the input's coverage block, verbatim. READ THIS FIRST. */ },
    "counts": {                 // what the run saw and did
      "things": 12, "links": 15, "findings": 17, "gaps": 2,
      "shortlisted": 4,         // candidates that survived selection
      "blocked": 0,             // ...of which this many were structurally invalid
      "model_calls": 0,         // 0 on a --dry-run
      "verification_verified": 0, "verification_qualified": 4,
      "verification_unverified": 0, "verification_rejected": 0
    },
    "considered": 4             // how many this one beat. 1-of-1 and 1-of-40 are
                                // different claims; the reader is told which
  },

  "hypothesis": {
    "id": "H-analog-t1-t11-via-t2",
    "motif": "analogical_transfer",
    // gap_closure | transitive_chain | analogical_transfer | condition_split
    "subject": "t1", "object": "t11",          // thing ids from the input graph
    "subject_name": "pirfenidone",             // resolved names, so a reader
    "object_name": "systemic sclerosis ILD",   // never has to join by hand
    "hops": 1,
    "tags": [],

    // The walk. On an analogical_transfer this is the DONOR's bridge edge --
    // the edge the analogue has and the subject lacks -- not a path from
    // `subject`. See note 8.
    "path": [{
      "link": "L8", "from": "t2", "from_name": "nintedanib",
      "how": "slows", "to": "t11", "to_name": "systemic sclerosis ILD",
      "reversed": false,        // true = walked against the stated direction
      "state": "single_source",
      "support": 0.711          // RECOMPUTED, not the link's stated confidence
    }],

    // A VECTOR, NOT A RANKING. Support and novelty are separate axes on
    // purpose: averaging them puts textbook statements first -- see note 7.
    "scores": {
      "support": 0.711,             // how well the graph backs it
      "novelty": 0.341,             // what the graph does NOT already state
      "testability": 0.7,
      "contradiction_risk": 0.0,    // findings arguing against the walk
      "structure": 1.0,             // motif strength, hub-damped
      "absence_reliability": 0.408  // how much the novelty claim is entitled to
    },
    "rank_score": 0.4607,           // the scalar that ordered selection

    // THE MODEL'S ENTIRE WORLD. Every id a claim may cite appears here, and
    // every id here appears in the input graph. See note 9.
    "evidence": {
      "links":    { "L8": { "from": "t2", "to": "t11", "how": "slows",
                            "state": "single_source", "basis": "primary",
                            "yes": ["f9"], "no": [], "no_effect": [],
                            "stated_confidence": 0.8,      // what the graph said
                            "recomputed_support": 0.711,   // what we make of it
                            "conditions": ["phase 3 trial"] } },
      "findings": { "f9": { "quote": "nintedanib reduced the annual rate of FVC decline",
                            "paper": "p9", "says": "yes", "where": "phase 3 trial",
                            "is_own_result": true, "hedged": false } },
      "papers":   { "p9": { "first_author": "Distler", "year": 2019,
                            "study_type": "clinical_trial", "is_preprint": false } },
      "things":   { "t1": { "name": "pirfenidone", "kind": "small_molecule" } },
      "gap":      null          // the gap row, on a gap_closure hypothesis
    },

    // Things true of the RUN that bear on this claim. Rendered once, not per
    // sentence, but never dropped.
    "caveats": [
      "The search is a sample, not the literature: 36 of 318 results were read.",
      "Novelty rests on a gap; this graph is not entitled to strong absence claims."
    ],

    // WRITTEN BY THE MODEL. null on a --dry-run, and null if the deterministic
    // gates rejected the candidate before a call was worth spending.
    "articulation": {
      "statement": "...",            // the hypothesis in one testable sentence
      "mechanism": "...",            // the causal chain, in graph terms
      "claims": [{                   // decomposed into separately checkable pieces
        "text": "pirfenidone inhibits TGF-beta1 signalling",
        "cites": ["L1", "f2"],       // PACK IDS ONLY -- anything else is rejected
        "inferred": false            // true = a step of reasoning, not a graph fact
      }],
      "novel_because": "...",        // what the graph does NOT already state
      "predictions": ["..."],
      "falsifier": "...",            // the single observation that would kill it
      "decisive_experiment": "...",  // the cheapest discriminating experiment
      "assumptions": ["..."]         // what must hold but is not in the graph
    },

    // THE ADVERSARIAL PASS. Each critic gets a different lens, because three
    // identical refuters mostly agree -- see note 10.
    "critiques": [{
      "verdict": "partly_supported",
      // supported | partly_supported | unsupported | contradicted
      "strongest_objection": "...",
      "unsupported_leaps": ["..."],
      "per_claim": [{ "claim_index": 0, "verdict": "supported",
                      "reason": "...", "cites": ["f2"] }],
      "alternative_explanation": "",  // a duller reading of the same evidence
      "lens": "mechanism"             // set by the harness, not the model
    }],
    "verdict": "partly_supported",    // critics' consensus, per refute_threshold.
                                      // null = the critics never ran

    // THE WHOLE PROCESS -- distinct from `verdict` above, which is only the
    // critics. Six gates in cost order. A SKIP IS NOT A PASS -- see note 11.
    "verification": {
      "verdict": "qualified",   // verified | qualified | unverified | rejected
      "gates": [
        { "name": "structure",     "status": "pass",
          "summary": "1 hop(s), path intact, not already stated",
          "issues": [], "halting": true },
        { "name": "independence",  "status": "warn",
          "summary": "all primary evidence here is from Distler; nothing replicates it",
          "issues": [{ "code": "single_group", "detail": "...",
                       "severity": "warning" }],
          "halting": true }
        // status: pass | warn | fail | skip
      ],
      "halted_at": null         // non-null = every gate below it DID NOT RUN
    },

    "elo": null,                // set only when the tournament ran
    "evolved_from": null,       // the id this was revised from, if it was
    "evolution_operator": null,

    // severity "error" means THIS IS NOT PRESENTABLE AS A FINDING.
    "issues": [{ "code": "single_group", "detail": "...", "severity": "warning" }],

    "asks": [ /* same shape as the document-level asks below */ ],
    "provenance": "analogical_transfer over g_demo1@round2 via L8"
  },

  // THE LOOP CLOSING. The exact request that would move this hypothesis, keyed
  // by id, in the graph builder's own request shape -- no prose to interpret.
  "asks": [{
    "graph_id": "g_demo1",
    "ask": "test_gap",          // expand_node | resolve_link | test_gap | new_question
    "target": "g1",             // an id in the input graph
    "depth": "deep",            // quick | standard | deep | exhaustive
    "reason": "novelty rests on this gap and nobody has searched for it",
    "for_hypothesis": "H-analog-t1-t11-via-t2"
  }]
}
```

### Reading order for a consumer

1. **`schema_version`** — know what you are parsing.
2. **`provenance.coverage`** — how hard the graph was searched. Read this
   *before* the scores. Absence of evidence is not evidence of absence.
3. **`hypothesis.verification.verdict`**, and `halted_at` with it. A verdict of
   `unverified` with `halted_at` set means the gates below the halt **did not
   run**. They are not passes.
4. **`hypothesis.issues`** — anything at severity `error` means this is not
   presentable as a finding.
5. Only then, the claim itself.

---

## Worked example — a real run

Everything below is copied out of
[`examples/hypothesis.json`](../examples/hypothesis.json), produced by:

```bash
hypgen --graph examples/knowledge-graph.json --profile repurposing --dry-run --out examples/
```

Values are unedited. `articulation`, `critiques` and `verdict` are `null` or
empty because `--dry-run` makes no model calls — the deterministic half of the
pipeline produced everything shown, without an API key.

**What the run chose, and from what:**

```json
{
  "schema_version": "2.0",
  "provenance": {
    "graph_id": "g_demo1",
    "round": 2,
    "question": "can an existing small molecule be repositioned against idiopathic pulmonary fibrosis?",
    "generated_at": "2026-08-15T09:12:00Z",
    "coverage": {
      "depth": "deep", "found": 318, "read": 36, "used": 33,
      "truncated": true, "no_quote_discarded": 4,
      "limits": { "max_papers": 50, "max_queries": 6, "hit_limit": "max_papers" }
    },
    "counts": {
      "things": 12, "links": 15, "findings": 17, "gaps": 2,
      "shortlisted": 4, "blocked": 0, "model_calls": 0,
      "verification_verified": 0, "verification_qualified": 4,
      "verification_unverified": 0, "verification_rejected": 0
    },
    "considered": 4
  }
}
```

Read `coverage` first and the shape of the answer is already set: 36 of 318
results were read and the search truncated, so `absence_reliability` is
0.408 — this graph is entitled to a weak absence claim, not a strong one.

**The hypothesis, and its walk:**

```json
{
  "id": "H-analog-t1-t11-via-t2",
  "motif": "analogical_transfer",
  "subject": "t1", "object": "t11",
  "subject_name": "pirfenidone",
  "object_name": "systemic sclerosis ILD",
  "hops": 1,
  "path": [
    { "link": "L8", "from": "t2", "from_name": "nintedanib",
      "how": "slows", "to": "t11", "to_name": "systemic sclerosis ILD",
      "reversed": false, "state": "single_source", "support": 0.711 }
  ],
  "scores": {
    "support": 0.711, "novelty": 0.341, "testability": 0.7,
    "contradiction_risk": 0.0, "structure": 1.0, "absence_reliability": 0.408
  },
  "rank_score": 0.4607
}
```

The proposal is *pirfenidone → systemic sclerosis ILD*. The drawn edge `L8` is
**nintedanib → SSc-ILD** — the analogue's, not the subject's. That is what an
`analogical_transfer` is: the two molecules share neighbours, one has an edge
the other lacks, and the hypothesis is that the other has it too. The edge the
graph does *not* contain is the whole claim (note 8).

**The evidence pack — everything the model would be allowed to cite:**

```json
{
  "links": {
    "L8": { "from": "t2", "to": "t11", "how": "slows",
            "state": "single_source", "basis": "primary",
            "yes": ["f9"], "no": [], "no_effect": [],
            "stated_confidence": 0.8, "recomputed_support": 0.711,
            "conditions": ["phase 3 trial"] }
  },
  "findings": {
    "f9": { "from": "t2", "to": "t11", "how": "slows", "says": "yes",
            "quote": "nintedanib reduced the annual rate of FVC decline in SSc-ILD",
            "paper": "p9", "where": "phase 3 trial",
            "is_own_result": true, "hedged": false, "confidence": 0.95 }
  },
  "papers": {
    "p9": { "title": "Nintedanib slows decline in SSc-ILD", "year": 2019,
            "journal": "N Engl J Med", "doi": "10.1056/x9",
            "first_author": "Distler", "study_type": "clinical_trial",
            "is_preprint": false }
  }
}
```

The graph stated `confidence.overall: 0.8` on `L8`. Recomputed from `f9` + `p9`
it is **0.711** — a phase 3 trial, unhedged, first-hand, but from a single
research group. The stated number and the recomputed one are both in the file,
so a consumer can see exactly where this app disagreed with its input (note 3).

**Verification — the gate table, and what a `warn` costs:**

```json
{
  "verdict": "qualified",
  "gates": [
    { "name": "structure", "status": "pass",
      "summary": "1 hop(s), path intact, not already stated", "halting": true },
    { "name": "citations", "status": "skip", "summary": "not articulated",
      "halting": true },
    { "name": "consistency", "status": "skip", "summary": "not articulated",
      "halting": false },
    { "name": "independence", "status": "warn",
      "summary": "all primary evidence here is from Distler; nothing replicates it",
      "issues": [{ "code": "single_group",
                   "detail": "all primary evidence here is from Distler; nothing replicates it",
                   "severity": "warning" }],
      "halting": true }
  ],
  "halted_at": null
}
```

`citations` and `consistency` are `skip` with the reason stated — *not
articulated*, because no model ran. Four of six gates skipped is why the verdict
is `qualified` rather than `verified`: the process is reported as what it was,
never as more than it was.

The `independence` warning is the interesting one. Every piece of primary
evidence traces to `p9`, first author Distler. Under `--profile conservative`
this same graph produces a **fail** and a halt, because that profile requires
two independent groups. Same evidence, different stance, different verdict —
and the stance is in `provenance.params` so the difference is visible rather
than mysterious.

---

## What the core guarantees

Enforced in code, not merely intended. A consumer can build on these.

- **Every cited id was in the evidence pack.** A model that cites `L7` when
  `L7` was never shown to it has stopped reporting and started remembering. The
  citation is rejected and the hypothesis flagged with `illegal_citation`
  before it can reach the file.
- **Every pack id is an input-graph id.** The pack is assembled from the graph,
  so the chain from claim → finding → verbatim sentence → paper never leaves
  the input.
- **Support is recomputed, never inherited.** `links.confidence.overall` is
  read as a claim. `path[].support` and `evidence.links[].recomputed_support`
  come from `findings` + `papers` with study type, hedging, secondhand
  citation, preprint status and independent-group counts applied.
- **Novelty from a gap is scaled by `absence_reliability`**, and at `quick`
  depth that factor is zero. Nothing can claim to be new merely because this
  search did not surface it.
- **A skipped gate is never recorded as a passed one.** When a gate halts, every
  gate below it is written as `skip` with the halt named in `halted_at`.
- **The output is reproducible.** `provenance.params` is the complete resolved
  parameter set; the same graph and the same params produce the same
  hypothesis.
- **The document is self-contained.** Nothing in it requires re-reading the
  graph to be understood, which is what lets every adapter be a pure function
  of this one file.

---

## Notes

1. **`coverage` is not metadata, it is the denominator.** `absence_reliability`
   is `[0.0, 0.45, 0.8, 1.0]` by depth tier, ×0.6 if `truncated`, ×0.85 if
   `limits.hit_limit` is set. It multiplies every novelty score that leans on a
   gap. A graph at `quick` depth contributes zero absence evidence no matter
   what it does not contain.
2. **`searched_in_round: null` means nobody looked.** A gap that has been
   searched for and not found is a far stronger novelty claim than one nobody
   queried — and it is the difference between "unexplored" and "unread". This
   is why `test_gap` is the most common ask this app emits.
3. **`links.confidence` is a claim, not a fact.** This app recomputes support
   and reports both numbers, so where it disagrees with its input is visible
   rather than silent. If you disagree with *our* weights, `provenance.params`
   has all of them and `evidence` has the raw rows.
4. **`state: "disagreed"` usually is not a conflict.** Compare `where` on both
   camps first; different experimental conditions is the common case. That is
   why reconciling one is the `condition_split` motif rather than a data
   quality complaint.
5. **Hubs are damped, not banned.** Degree-weighted path counts (Rephetio's
   DWPC) stop "aspirin → inflammation → everything" from winning every run.
   `things.mentions` and node degree both feed it.
6. **Provenance travels with the claim.** Support 0.5 from a cautious run and
   support 0.5 from an ambitious one are different statements about the world,
   and the number alone cannot tell them apart. `provenance.params.stance` says
   which was asked for; `provenance.considered` says how many candidates this
   one beat.
7. **`scores` is a vector; `rank_score` is one collapse of it.** Support and
   novelty are deliberately separate axes — a fully supported hypothesis is a
   known fact, so averaging the two ranks textbook statements first. Re-rank
   with your own weights if the run's collapse is not the one you want.
8. **On an `analogical_transfer`, `path` is the donor's edge.** The subject
   reaches the object only by the proposed edge, which by construction the
   graph does not contain. Walking `path` from `subject` will not resolve, and
   a consumer that assumes it does will mark every analogical hypothesis broken
   — this exact bug shipped once and is now a regression test.
9. **`evidence` is the model's entire world.** It is assembled before any call
   and is the only thing the model sees, which is what makes the citation rule
   enforceable rather than aspirational.
10. **Critics get lenses, not copies.** Three identical refuters mostly agree; a
    mechanism critic and an evidence critic fail on different things. `lens` is
    set by the harness so a consumer can see which angles were actually tried.
11. **A skip is not a pass.** Gates run cheapest-first so the four deterministic
    ones can reject a candidate before the adversarial gate spends a call. Five
    green checks because the sixth never ran would read as more verified than
    the truth, so the halt is named and everything below it is `skip`.
12. **`--dry-run` is a real run.** It executes the entire deterministic half —
    traversal, motifs, scoring, selection, evidence packs, and four of the six
    gates — and writes a complete, valid `hypothesis.json` with `articulation`
    null. It needs no API key, and it is the right way to see whether a graph
    supports anything before paying for the model stages.
