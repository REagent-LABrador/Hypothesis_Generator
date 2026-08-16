"""Multi-objective scoring, recomputed from findings rather than trusted.

The graph builder ships a `links.confidence` block, and schema note 3 explicitly invites
a consumer to recompute it with its own weights. We do, for two reasons: the
weights are a judgement call belonging to whoever acts on the output, and
recomputing forces us to touch every finding and paper, which is what populates
the audit trail.

The output is a vector, not a number. Support and novelty are separate axes and
are never averaged before ranking -- a hypothesis with support 1.0 and novelty
0.0 is a textbook fact, and any scalar that ranks it first is measuring the
wrong thing. ``rank_score`` exists only to order a list on a page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hyp_gen.generate.candidates import Candidate
from hyp_gen.graph import GraphIndex, Link
from hyp_gen.params import Params


@dataclass
class LinkSupport:
    """Our own read of how well one link is evidenced."""

    link_id: str
    evidence_quality: float
    agreement: float
    independence: float
    support: float
    stated_overall: float
    yes_count: int
    no_count: int
    groups: int = 0
    capped: bool = False
    paper_ids: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)

    @property
    def drift(self) -> float:
        """Ours minus the graph builder's. Large drift either way is worth a look."""
        return round(self.support - self.stated_overall, 3)


def _recency(year: int | None, latest: int | None, half_life: float) -> float:
    """Exponential decay on age, relative to the newest paper in the graph.

    Relative rather than absolute because the graph carries no clock and we do
    not want a run's output to change with the wall date. Off by default: decay
    is right for a fast-moving target and wrong for settled biochemistry, and
    guessing which is which is not our call.
    """
    if not half_life or year is None or latest is None:
        return 1.0
    age = max(latest - year, 0)
    return 0.5 ** (age / half_life)


def score_link(index: GraphIndex, link: Link, params: Params) -> LinkSupport:
    e = params.evidence
    findings = index.findings_for(link)
    latest = max((p.year for p in index.papers.values() if p.year), default=None)

    qualities: list[float] = []
    yes_w = no_w = 0.0
    groups: set[str] = set()
    papers: list[str] = []

    for finding in findings:
        paper = index.paper_for(finding)
        weight = e.study_weights.get(
            paper.study_type if paper else "computational", 0.3
        )
        if finding.hedged:
            weight *= e.hedged_penalty
        if not finding.is_own_result:
            weight *= e.secondhand_penalty
        if paper is not None:
            if paper.is_preprint:
                weight *= e.preprint_penalty
            weight *= _recency(paper.year, latest, e.recency_half_life)
        # The extractor's own read-accuracy score gates the finding: a
        # confidently misread sentence is worse than a hedged accurate one.
        weight *= max(finding.confidence, 0.0)
        qualities.append(weight)

        if finding.says == "yes":
            yes_w += weight
        elif finding.says == "no":
            no_w += weight

        if paper is not None:
            if paper.id not in papers:
                papers.append(paper.id)
            if finding.is_own_result and paper.first_author:
                groups.add(paper.first_author)

    evidence_quality = max(qualities) if qualities else 0.0
    evidence_quality *= e.basis_penalty.get(link.basis, 1.0)

    total = yes_w + no_w
    # No signal either way collapses to 0.5 rather than 1.0: an unopposed claim
    # nobody has tested is not agreement.
    agreement = (yes_w / total) if total > 0 else 0.5
    independence = 0.0 if len(groups) <= 1 else 1.0 - 1.0 / len(groups)

    w = e.support_weights
    support = (
        w["evidence_quality"] * evidence_quality
        + w["agreement"] * agreement
        + w["independence"] * independence
    )

    # One lab reporting a result five times is one result. The cap is the
    # difference between "replicated" and "repeated".
    capped = len(groups) < e.min_independent_groups and support > e.single_group_cap
    if capped:
        support = e.single_group_cap

    return LinkSupport(
        link_id=link.id,
        evidence_quality=round(evidence_quality, 3),
        agreement=round(agreement, 3),
        independence=round(independence, 3),
        support=round(support, 3),
        stated_overall=link.confidence.overall,
        yes_count=len(link.yes),
        no_count=len(link.no),
        groups=len(groups),
        capped=capped,
        paper_ids=papers,
        conditions=index.conditions_for(link),
    )


@dataclass
class Scores:
    support: float
    novelty: float
    testability: float
    contradiction_risk: float
    structure: float
    absence_reliability: float
    rank_score: float = 0.0
    structure_raw: float = 0.0
    per_link: list[LinkSupport] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def vector(self) -> dict[str, float]:
        return {
            "support": self.support,
            "novelty": self.novelty,
            "testability": self.testability,
            "contradiction_risk": self.contradiction_risk,
            "structure": self.structure,
            "absence_reliability": self.absence_reliability,
        }


def _aggregate(values: list[float], how: str) -> float:
    if not values:
        return 0.0
    if how == "mean":
        return sum(values) / len(values)
    if how == "noisy_or":
        # Correct for several independent sources of support for the *same*
        # conclusion; wrong for a chain, where every link must hold. Only sane
        # at max_hops=1, which is why it is not the default.
        product = 1.0
        for v in values:
            product *= 1.0 - min(max(v, 0.0), 1.0)
        return 1.0 - product
    return min(values)


def score_candidate(index: GraphIndex, candidate: Candidate, params: Params) -> Scores:
    ev, nov, sel = params.evidence, params.novelty, params.selection
    notes: list[str] = []

    per_link = [
        score_link(index, index.links[lid], params)
        for lid in candidate.link_ids
        if lid in index.links
    ]
    support = _aggregate([l.support for l in per_link], ev.chain_aggregation)
    if any(l.capped for l in per_link):
        notes.append("support capped: a link rests on a single research group")

    # -- novelty ---------------------------------------------------------
    # Distance from what is already stated, discounted by how entitled this
    # graph is to claim absence at all.
    reliability = index.absence_reliability()
    scale = (0.35 + 0.65 * reliability) if nov.respect_absence_reliability else 1.0

    novelty = nov.hop_novelty * max(candidate.hops - 1, 0)
    if candidate.gap_id:
        gap = index.gaps.get(candidate.gap_id)
        bonus = nov.gap_novelty_bonus
        if gap is not None and gap.searched_in_round:
            # Looked for and not found is a far stronger claim than never
            # looked for.
            bonus += nov.searched_gap_bonus
            notes.append(f"{gap.id} was searched in round {gap.searched_in_round}")
        if gap is not None:
            # Mirror the graph builder's own cap: a gap is a proposal, not a finding, so
            # we must not out-confide our input.
            bonus = min(bonus, nov.gap_confidence_cap)
        novelty += bonus
    if candidate.motif == "analogical_transfer":
        novelty += 0.15
    if candidate.motif == "condition_split":
        # Not a new connection at all: a reinterpretation of a stated one.
        novelty *= 0.5

    # LLM novelty judges reliably over-reward densely-connected concepts -- the
    # famous, safe pairing scores well and discovers nothing. Correct for it
    # before any model sees the candidate.
    degrees = [index.degree(candidate.subject), index.degree(candidate.object)]
    busiest = max(index.degree(t) for t in index.things) if index.things else 1
    popularity = (sum(degrees) / 2) / max(busiest, 1)
    novelty -= nov.popularity_penalty * popularity
    novelty = max(0.0, min(novelty, 1.0)) * scale

    # -- testability ------------------------------------------------------
    kinds = {index.kind(n) for n in candidate.node_ids()}
    testability = 0.3
    if kinds & set(sel.testable_kinds):
        testability += 0.4
    if candidate.motif == "condition_split" and len(candidate.conditions) > 1:
        testability += 0.3  # a named pair of conditions is a ready-made experiment
    if candidate.hops > 2:
        testability -= 0.15 * (candidate.hops - 2)
    testability = max(0.0, min(testability, 1.0))

    # -- contradiction risk ----------------------------------------------
    risky = 0.0
    for lid in candidate.link_ids:
        link = index.links.get(lid)
        if link is None:
            continue
        if link.state == "disagreed" or link.no:
            risky += 1
        if link.basis in ("hedged_only", "background_only"):
            risky += 1
    denom = max(len(candidate.link_ids), 1)
    contradiction_risk = min((risky / denom) * ev.contradiction_weight, 1.0)
    if candidate.motif == "condition_split":
        contradiction_risk *= 0.4  # disagreement is the premise, not a defect
    if "reversed_edge" in candidate.tags:
        contradiction_risk = min(contradiction_risk + 0.2, 1.0)
        notes.append("chain crosses at least one link against its stated direction")

    return Scores(
        support=round(support, 3),
        novelty=round(novelty, 3),
        testability=round(testability, 3),
        contradiction_risk=round(contradiction_risk, 3),
        structure=0.0,  # set in normalise(), which needs the whole set
        structure_raw=candidate.weight,
        absence_reliability=reliability,
        per_link=per_link,
        notes=notes,
    )


def normalise(scored: list[tuple[Candidate, Scores]], params: Params) -> None:
    """Fill in ``structure`` and ``rank_score``. Mutates in place.

    DWPC is only meaningful relative to the other paths in the same graph --
    the absolute number depends on graph density and means nothing on its own
    -- so this needs the whole set and cannot happen in ``score_candidate``.
    """
    weights = params.selection.rank_weights
    priors = params.motifs.weights
    biggest = max((s.structure_raw for _, s in scored), default=0.0)

    for candidate, s in scored:
        s.structure = round(s.structure_raw / biggest, 3) if biggest > 0 else 0.0
        base = (
            weights.get("support", 0.0) * s.support
            + weights.get("novelty", 0.0) * s.novelty
            + weights.get("testability", 0.0) * s.testability
            + weights.get("contradiction_risk", 0.0) * s.contradiction_risk
            + weights.get("structure", 0.15) * s.structure
        )
        # The motif prior multiplies, per its definition in params. Only the
        # display order depends on this; every axis is reported separately.
        s.rank_score = round(base * priors.get(candidate.motif, 1.0), 4)


def score_all(
    index: GraphIndex, candidates: list[Candidate], params: Params
) -> list[tuple[Candidate, Scores]]:
    scored = [(c, score_candidate(index, c, params)) for c in candidates]
    normalise(scored, params)
    scored.sort(key=lambda pair: pair[1].rank_score, reverse=True)
    return scored
