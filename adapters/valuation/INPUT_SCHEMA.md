# INPUT — one or more `hypothesis.json`

The valuation adapter turns hypothesis documents into program briefs for the ROI model.

The input contract is the core's output contract, unchanged:
[`src/hyp_gen/OUTPUT_SCHEMA.md`](../../src/hyp_gen/OUTPUT_SCHEMA.md).

This adapter accepts **one or more** documents. The core emits one per run, so
several means several runs; each document carries its own provenance, and
`adapters.common.load()` refuses to bundle documents from different graphs —
their scores are not comparable and a shared header would claim they are.

It also requires a **`ProgramFrame`** — the analyst's half of the input, which no graph can supply. Its schema is in OUTPUT_SCHEMA.md beside this file.

What this adapter may not do, like every adapter: read the knowledge graph,
call a model, or state anything the documents do not carry.
