"""The only place a model is called.

    llm     the Judge client -- retries, budget accounting, refusal handling
    reason  articulate -> critique from N lenses -> compare -> evolve

Reached only by candidates that survived ``generate.select``, so cost scales
with ``selection.top_k`` rather than with graph size. Nothing produced here is
trusted on its own: it all goes to ``hyp_gen.checks`` before it can reach the
record.
"""
