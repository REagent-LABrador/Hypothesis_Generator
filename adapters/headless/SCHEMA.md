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

`LIVE` always uses the provider-backed Judge. `REPLAY` uses only the deterministic
local pipeline and is marked `DETERMINISTIC_REPLAY`; neither mode falls back to
the other.

The output always contains `status`, `execution_mode`, `output_origin`,
`hypothesis`, `cards`, `roi_request`, and `error`. `hypothesis` is the canonical
single `HypothesisDocument`. `cards` is the existing WebPayload with required
top-level interpretability. `roi_request`, when present, is the complete rNPV
module v1 request. A downstream-only failure preserves any hypothesis and cards
already produced.
