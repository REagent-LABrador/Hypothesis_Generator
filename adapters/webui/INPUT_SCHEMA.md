# INPUT — one or more `hypothesis.json`

The webui adapter turns hypothesis documents into a card payload and an SVG of the graph walks.

The input contract is the core's output contract, unchanged:
[`src/hyp_gen/OUTPUT_SCHEMA.md`](../../src/hyp_gen/OUTPUT_SCHEMA.md).

This adapter accepts **one or more** documents. The core emits one per run, so
several means several runs; each document carries its own provenance, and
`adapters.common.load()` refuses to bundle documents from different graphs —
their scores are not comparable and a shared header would claim they are.


What this adapter may not do, like every adapter: read the knowledge graph,
call a model, or state anything the documents do not carry.
