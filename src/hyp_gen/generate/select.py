"""Which candidates survive to the expensive stages.

Ranking by score alone returns twelve versions of one idea. The graph's densest
neighbourhood wins every slot, and the record looks confident and says one thing.
Maximal Marginal Relevance is the standard fix: each pick is scored on what it
adds *given what is already picked*, with a single lambda trading relevance
against redundancy.

Order of operations matters and is deliberate:

    thresholds -> Pareto front (optional) -> MMR -> quotas -> top_k

Thresholds first because they are the caller's hard "not worth reading". The
Pareto front next, because it is a statement about the whole set. MMR last among
the scoring steps, because it is the only one that depends on what has already
been chosen. Quotas are applied during MMR selection rather than after, so a
rejected pick frees its slot for the next-best *different* idea instead of
shrinking the record.
"""

from __future__ import annotations

from hyp_gen.generate.candidates import Candidate
from hyp_gen.params import Params
from hyp_gen.generate.scoring import Scores

PARETO_AXES = ("support", "novelty", "testability")

Pair = tuple[Candidate, Scores]


def pareto_front(scored: list[Pair]) -> list[Pair]:
    """Non-dominated set over support / novelty / testability.

    A weighted sum picks one taste and hides it in a constant. The front hands
    the trade-off to whoever consumes the record: ROI weighs novelty and cost
    differently than a validation team weighs support.
    """
    front: list[Pair] = []
    for cand, scores in scored:
        dominated = any(
            all(getattr(other, a) >= getattr(scores, a) for a in PARETO_AXES)
            and any(getattr(other, a) > getattr(scores, a) for a in PARETO_AXES)
            for _, other in scored
        )
        if not dominated:
            front.append((cand, scores))
    return front


def similarity(a: Candidate, b: Candidate, how: str) -> float:
    """How redundant two candidates are, in [0, 1]."""
    if how == "motif":
        return 1.0 if a.motif == b.motif else 0.0
    if how == "endpoint":
        ends_a = {a.subject, a.object}
        ends_b = {b.subject, b.object}
        return len(ends_a & ends_b) / len(ends_a | ends_b)
    # jaccard_nodes: two hypotheses over the same things are the same idea told
    # twice, even when the motifs differ.
    na, nb = set(a.node_ids()), set(b.node_ids())
    union = na | nb
    return len(na & nb) / len(union) if union else 0.0


def _quota_ok(candidate: Candidate, chosen: list[Candidate], params: Params) -> bool:
    s = params.selection
    if s.max_per_subject and sum(
        1 for c in chosen if c.subject == candidate.subject
    ) >= s.max_per_subject:
        return False
    if s.max_per_object and sum(
        1 for c in chosen if c.object == candidate.object
    ) >= s.max_per_object:
        return False
    if s.max_per_motif and sum(
        1 for c in chosen if c.motif == candidate.motif
    ) >= s.max_per_motif:
        return False
    return True


def mmr(scored: list[Pair], params: Params) -> list[Pair]:
    """Greedy MMR with quotas, highest marginal value first."""
    s = params.selection
    lam = s.diversity_lambda
    pool = list(scored)
    chosen: list[Pair] = []

    while pool and len(chosen) < s.top_k:
        best: Pair | None = None
        best_value = float("-inf")
        for pair in pool:
            candidate, scores = pair
            if not _quota_ok(candidate, [c for c, _ in chosen], params):
                continue
            redundancy = max(
                (similarity(candidate, other, s.similarity) for other, _ in chosen),
                default=0.0,
            )
            value = lam * scores.rank_score - (1.0 - lam) * redundancy
            # Ties broken by id so a run is reproducible without a seed.
            if value > best_value or (
                value == best_value and best is not None and candidate.id < best[0].id
            ):
                best, best_value = pair, value
        if best is None:
            break  # everything left is quota-blocked
        chosen.append(best)
        pool.remove(best)
    return chosen


def select(scored: list[Pair], params: Params) -> list[Pair]:
    s = params.selection
    kept = [
        (c, sc)
        for c, sc in scored
        if sc.support >= s.min_support
        and sc.novelty >= s.min_novelty
        and sc.contradiction_risk <= s.max_contradiction_risk
    ]
    if s.require_pareto:
        kept = pareto_front(kept)
    kept.sort(key=lambda pair: pair[1].rank_score, reverse=True)
    return mmr(kept, params)
