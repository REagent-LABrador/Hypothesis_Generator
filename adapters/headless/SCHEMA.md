# Headless adapter — Input and Output Contract

`hypgen-run` composes one focused core run with the existing cards and valuation
adapters. It reads one request file and writes one result file.

```bash
hypgen-run --mode live --input request.json --output result.json
hypgen-run --mode replay --input request.json --output result.json
```

The request requires the native `graph`, one `focus_thing_id`, an explicit
`profile`, a complete valuation `ProgramFrame`, and `roi` settings containing
the request id, comparables, simulation count and seed. Nothing chooses a
program or fills the four required valuation years.

Each focused invocation gives its canonical hypothesis a deterministic,
focus-scoped id. The document asks, card, interpretability headline, ROI
evidence, and ROI assumptions all use that same id, so two branches that select
the same structural candidate remain distinct downstream. This scoping belongs
only to the headless focused boundary; the core generator's unfocused ids are
unchanged.

When the focus is a biomarker or process rather than a disease, the explicit
frame must also name `target`, `modality`, `therapeutic_area`, and a non-placeholder
`target_population`. The hypothesis supplies mechanism evidence only. The frame's
`target_population` is the economic population label, and the frame target is a
target/program placeholder—not a discovered or nominated molecule. Missing
fields return `ROI_FRAME_INCOMPLETE`; no graph process is relabelled as a disease.

`LIVE` always uses the provider-backed Judge. `REPLAY` uses only the deterministic
local pipeline and is marked `DETERMINISTIC_REPLAY`; neither mode falls back to
the other.

The output always contains `status`, `execution_mode`, `output_origin`,
`hypothesis`, `cards`, `roi_request`, and `error`. `hypothesis` is the canonical
single `HypothesisDocument`. `cards` is the existing WebPayload with required
top-level interpretability. `roi_request`, when present, is the complete rNPV
module v1 request. A downstream-only failure preserves any hypothesis and cards
already produced.
