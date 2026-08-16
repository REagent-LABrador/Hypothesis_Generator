# INPUT — one or more `hypothesis.json`

The report adapter turns hypothesis documents into markdown somebody reads.

The input contract is the core's output contract, unchanged:
[`src/hyp_gen/OUTPUT_SCHEMA.md`](../../src/hyp_gen/OUTPUT_SCHEMA.md).

This adapter accepts **one or more** documents. The core emits one per run, so
several means several runs; each document carries its own provenance, and
`adapters.common.load()` refuses to bundle documents from different graphs —
their scores are not comparable and a shared header would claim they are.

`table` and `trace` mode are the ones that gain from several documents: one row each, and shared nodes drawn once.

What this adapter may not do, like every adapter: read the knowledge graph,
call a model, or state anything the documents do not carry.
