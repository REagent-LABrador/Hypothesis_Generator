"""Everything the model said, evaluated back against the graph.

    validate  structure against the graph, citations against the evidence pack
    verify    six gates in cost order; a halt skips the rest, loudly

The four deterministic gates run before the adversarial one, so a candidate can
be rejected without spending a model call. A skipped gate is never recorded as
a passed one -- the halt that caused it is named in the record.
"""
