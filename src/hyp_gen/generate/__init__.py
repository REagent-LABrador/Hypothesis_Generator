"""Deterministic pattern finding: the graph is the only input.

No model call happens here and no API key is needed, so this subpackage is
exactly what ``--dry-run`` runs. Its job is to turn a graph into a short,
ranked list of structural candidates, each already carrying the evidence a
model would need to write it up.

    candidates  four motifs -> structural candidates
    scoring     recompute support from findings; novelty, risk, testability
    select      thresholds -> Pareto front -> MMR -> quotas
    evidence    per-candidate pack: the model's entire world
    asks        weakest point -> one request back to the graph builder, by id

Same graph and same params in, same candidates out. That determinism is what
makes a disagreement about the output a disagreement about parameters.
"""
