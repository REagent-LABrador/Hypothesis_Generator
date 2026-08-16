"""The knobs. A hypothesis is a function of (graph, params) -- nothing else.

Two runs over the same graph with the same params produce the same candidates
and the same scores, which is what makes a disagreement about output a
disagreement about parameters rather than about luck. Only articulation,
critique and the tournament are model calls, and they are pinned to the
evidence the deterministic stages selected.

Every group below is named after the published method it comes from, so a
reviewer can check the default against the source rather than against taste:

- ``Framing``     Swanson's ABC model: open vs closed discovery.
- ``Traversal``   Rephetio/Hetionet: typed metapaths, degree-weighted paths.
- ``Evidence``    recomputed from `findings` + `papers`, not trusted from the graph builder.
- ``Novelty``     gap-based, discounted by how hard the graph builder actually looked.
- ``Selection``   MMR: relevance traded against redundancy by one lambda.
- ``Ranking``     co-scientist: pairwise debate, Elo, evolve.
- ``Loop``        when to go back to the graph builder for more graph.
- ``Budget``      hard ceilings, so a hairball graph cannot run away.

Profiles are plain JSON so a demo can show one graph yielding a conservative
and a speculative record side by side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# The motifs the enumerator knows how to find. Each one is a distinct reason a
# hypothesis could be worth stating -- see candidates.py for what each means.
MOTIFS = (
    "gap_closure",
    "transitive_chain",
    "analogical_transfer",
    "condition_split",
)

DiscoveryMode = Literal["open", "closed"]
ChainAggregation = Literal["weakest", "mean", "noisy_or"]


class FramingParams(BaseModel):
    """What question is being asked of the graph.

    Swanson's ABC model splits literature-based discovery in two, and the split
    is not cosmetic -- it changes what a good answer looks like. *Open*
    discovery fixes A and hunts for any C worth connecting to it; that is
    generation. *Closed* discovery fixes both A and C and hunts for the B that
    explains them; that is mechanism-finding, and it is what you run when a
    clinical prompt already names both ends.
    """

    model_config = ConfigDict(extra="forbid")

    mode: DiscoveryMode = "open"

    anchors: tuple[str, ...] = ()
    """Thing ids (or names/aliases) that every hypothesis must touch -- the A
    side. Empty means the whole graph seeds. In closed mode this must be
    non-empty."""

    targets: tuple[str, ...] = ()
    """The C side. Only meaningful in closed mode, where the output is the set
    of B terms bridging anchors to targets."""

    exclude: tuple[str, ...] = ()
    """Things to route around entirely -- usually the trivially-connecting hub
    that a clinician already knows about and does not want restated."""


class TraversalParams(BaseModel):
    """How far, how carefully, and along which typed shapes to walk.

    The two parameters that matter most here are ``max_hops`` and
    ``hub_damping``, and they pull against each other: more hops finds more,
    and almost all of what it finds is a path laundered through a promiscuous
    node. Rephetio's answer -- keep paths short (2-4) and downweight by node
    degree rather than banning hubs outright -- is the default here.
    """

    model_config = ConfigDict(extra="forbid")

    max_hops: int = 3
    """Longest chain of links a hypothesis may rest on. Every extra hop
    multiplies the ways the story can be wrong, so this is the single most
    important speculation dial. Rephetio evaluated metapaths of length 2-4 and
    found little signal past that."""

    min_link_confidence: float = 0.2
    """Links below this are not walked. Note this is the graph builder's arithmetic
    confidence, which we recompute ourselves in scoring -- this is a coarse
    pre-filter to keep traversal tractable, not the judgement."""

    allow_no_effect_edges: bool = False
    """A `no_effect` link is a measured non-relationship. Chaining through one
    usually produces nonsense, but it is occasionally the point (a negative
    result routing you to an alternative mechanism)."""

    allow_negative_edges: bool = False
    """Walk links whose findings are predominantly `no`. Off by default for the
    same reason: "A does not do B, B does C" composes to nothing."""

    allow_edge_reversal: bool = True
    """Cross a link against its stated direction. `binds` is near-symmetric,
    `inhibits` is not, so a reversed hop is a weaker claim rather than an equal
    one -- see ``reversal_penalty``. Turn this off for strictly causal graphs."""

    reversal_penalty: float = 0.7
    """Multiplier on a hop crossed backwards. Applied per reversed edge, so a
    chain that flips twice is punished twice."""

    max_paths_per_pair: int = 2
    max_branch_per_node: int = 12
    """Beam width. A node with 200 neighbours would otherwise dominate the
    frontier; the highest-confidence neighbours are kept."""

    # -- typed shape constraints (Rephetio metapaths) ---------------------

    seed_kinds: tuple[str, ...] = ()
    """Restrict which node kinds may start a path. Empty means all."""

    target_kinds: tuple[str, ...] = ()
    """Restrict which node kinds a path may end on. Empty means all. Setting
    this to ("disease",) is how you ask for therapeutic hypotheses only."""

    intermediate_kinds: tuple[str, ...] = ()
    """Kinds allowed in the middle of a chain. Empty means all. This is the
    cheap version of a metapath: it constrains the shape without enumerating
    every sequence."""

    metapaths: tuple[tuple[str, ...], ...] = ()
    """The expensive, precise version: explicit kind sequences a path must
    match, e.g. ``("small_molecule", "protein", "gene", "disease")`` -- the
    compound-binds-gene-associates-disease shape Rephetio found most
    predictive. Empty means any sequence the kind filters permit. When
    non-empty this *replaces* the kind filters."""

    predicates_allow: tuple[str, ...] = ()
    """`how` values a path may use. Empty means all. The graph builder's predicate
    vocabulary is open, so this is a denylist-first world in practice."""

    predicates_deny: tuple[str, ...] = ("correlates_with", "co_occurs_with")
    """Predicates that are real relationships but compose into gibberish. Two
    correlations in a row imply nothing, and this is the classic way an ABC
    pipeline produces confident nonsense."""

    # -- hub control (DWPC) -----------------------------------------------

    hub_damping: float = 0.4
    """The DWPC exponent `w`. Each edge on a path is divided by
    ``(degree(src) * degree(dst)) ** w``, so a path through a node everything
    touches contributes almost nothing. Rephetio tuned this to 0.4 across
    1206 metapaths; 0.0 disables damping and 1.0 is aggressive."""

    max_node_degree: int = 0
    """Hard ban on traversing through a node with more degree than this. 0
    means no ban -- prefer ``hub_damping``, which degrades rather than cliffs.
    Use this only when one node is a known annotation artifact."""

    min_mentions: int = 0
    """Ignore things mentioned in fewer than this many papers -- a floor on how
    real an entity has to be before it anchors a hypothesis."""

    min_dwpc: float = 0.0
    """Drop candidates whose degree-weighted path count falls below this. The
    principled successor to ``min_link_confidence`` for multi-hop chains: it
    judges the whole path, not each link in isolation."""

    max_candidates: int = 400
    """Hard stop on enumeration before scoring, so a hairball graph cannot
    blow up the run."""


class MotifParams(BaseModel):
    """Which structural reasons-to-speak are in play, and how much each counts."""

    model_config = ConfigDict(extra="forbid")

    enabled: tuple[str, ...] = MOTIFS

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "gap_closure": 1.00,
            "transitive_chain": 0.90,
            "analogical_transfer": 0.70,
            "condition_split": 0.85,
        }
    )
    """Prior on each motif, multiplied into the rank score. Analogical transfer
    is discounted because it reasons from similarity rather than from a path --
    it is the motif most likely to be fluent and wrong."""

    analogy_min_shared: int = 2
    """Neighbours two things must share before one is treated as a stand-in for
    the other."""

    analogy_min_jaccard: float = 0.15
    """Shared-neighbour count alone rewards hubs: two promiscuous nodes share
    many neighbours while being nothing alike. Jaccard normalises for that."""

    analogy_same_kind_only: bool = True
    """Analogy across kinds is usually a category error."""

    condition_split_requires_where: bool = True
    """Only propose a condition split when the disagreeing findings actually
    state different `where` values. Otherwise the hypothesis is "maybe it's
    conditions" with no candidate condition in hand."""

    require_unstated: bool = True
    """Drop any candidate whose endpoints already have a stated link. A
    hypothesis that restates a finding is a summary, not a hypothesis."""


class EvidenceParams(BaseModel):
    """Weights for the evidence arithmetic we recompute from findings.

    The graph builder ships a `links.confidence` block, and schema note 3 explicitly
    invites us to disagree with it. We do: these weights recompute support from
    `findings` + `papers` so the number that ranks a hypothesis is one we can
    defend line by line.
    """

    model_config = ConfigDict(extra="forbid")

    study_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "meta_analysis": 1.00,
            "clinical_trial": 0.95,
            "human_cohort": 0.80,
            "animal": 0.60,
            "test_tube": 0.45,
            "computational": 0.30,
            "review": 0.25,
        }
    )
    hedged_penalty: float = 0.75
    """Multiplier applied to a hedged finding ("may", "suggests")."""

    secondhand_penalty: float = 0.6
    """Multiplier when `is_own_result` is false -- the paper is citing someone
    else, so it is not an independent observation."""

    preprint_penalty: float = 0.85

    basis_penalty: dict[str, float] = Field(
        default_factory=lambda: {
            "primary": 1.0,
            "mixed": 0.9,
            "hedged_only": 0.7,
            "background_only": 0.5,
        }
    )

    recency_half_life: float = 0.0
    """Years after which a finding's weight halves. 0 disables decay. Non-zero
    is right for fast-moving targets and wrong for settled biochemistry, so it
    is off by default rather than guessed."""

    min_independent_groups: int = 1
    """Distinct `first_author` values a link needs before its support may
    exceed ``single_group_cap``. One lab reporting a result five times is one
    result."""

    single_group_cap: float = 0.6

    support_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "evidence_quality": 0.40,
            "agreement": 0.35,
            "independence": 0.25,
        }
    )

    chain_aggregation: ChainAggregation = "weakest"
    """How per-link support composes along a chain.

    ``weakest``  a chain is only as strong as its weakest link -- the honest
                 default, and the only one that cannot be gamed by padding.
    ``mean``     lets one strong link launder two weak ones.
    ``noisy_or`` treats links as independent evidence *for the same
                 conclusion*. Correct for converging support, wrong for a
                 chain, so it is only sane with ``max_hops=1``.
    """

    contradiction_weight: float = 1.0
    """How hard a `no` finding on any link of the chain pushes the
    contradiction-risk axis. Raise it when a wrong hypothesis is expensive."""


class NoveltyParams(BaseModel):
    """What makes a statement new, and how much of that to believe.

    The trap here is that absence in the graph is not absence in the
    literature. Schema note 2 is blunt: at `quick` depth, absence means
    unknown. Every bonus below is multiplied by the graph's own
    ``absence_reliability()``, so a shallow graph cannot mint novelty.
    """

    model_config = ConfigDict(extra="forbid")

    hop_novelty: float = 0.22
    """Novelty added per hop beyond the first."""

    gap_novelty_bonus: float = 0.25
    """Novelty added when the graph builder itself flagged the pair as a gap."""

    searched_gap_bonus: float = 0.20
    """Extra when that gap has `searched_in_round` set. A pair somebody looked
    for and did not find is a far stronger claim than one nobody searched --
    the schema says so, and this is where that pays off."""

    gap_confidence_cap: float = 0.6
    """The graph builder caps gap confidence at 0.6 because a gap is a proposal, not a
    finding. Mirrored here so we cannot out-confide our own input."""

    popularity_penalty: float = 0.15
    """Novelty subtracted in proportion to how well-connected the endpoints
    already are. LLM novelty judges reliably over-reward densely-connected
    concepts -- the safe, famous pairing scores well and discovers nothing --
    so the correction is applied before any model sees the candidate."""

    respect_absence_reliability: bool = True
    """Scale every gap-derived bonus by coverage depth and truncation. Turning
    this off is only defensible on an `exhaustive` graph."""


class SelectionParams(BaseModel):
    """Which candidates survive to the expensive stages.

    Ranking by score alone returns twelve versions of one idea, because the
    graph's densest neighbourhood wins every slot. MMR is the standard fix:
    each pick is scored against its marginal value given what is already
    picked, with one lambda trading relevance against redundancy.
    """

    model_config = ConfigDict(extra="forbid")

    top_k: int = 8
    """How many candidates survive to articulation. Model calls happen per
    survivor, so this is the cost dial."""

    diversity_lambda: float = 0.7
    """MMR lambda. 1.0 is pure score (redundant records), 0.0 is pure novelty
    against the selected set (incoherent records). 0.7 keeps the top pick and
    spends the tail on coverage."""

    similarity: Literal["jaccard_nodes", "endpoint", "motif"] = "jaccard_nodes"
    """How two candidates are judged redundant: overlap of the things they
    touch (default), sharing an endpoint, or sharing a motif."""

    max_per_subject: int = 2
    max_per_object: int = 2
    max_per_motif: int = 0
    """Hard quotas, applied after MMR. 0 disables. These exist because one
    fashionable target will otherwise take every slot even under MMR."""

    min_support: float = 0.0
    min_novelty: float = 0.0
    max_contradiction_risk: float = 1.0

    require_pareto: bool = False
    """Keep only the Pareto front over (support, novelty, testability).
    Ranking by a weighted sum quietly encodes one taste; the front does not."""

    rank_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "support": 0.35,
            "novelty": 0.35,
            "testability": 0.20,
            "contradiction_risk": -0.10,
        }
    )
    """Used only to order the shortlist for display. Support and novelty are
    reported separately and never collapsed before this point: a fully
    supported hypothesis is a known fact, and averaging the two axes ranks
    textbook statements first."""

    testable_kinds: tuple[str, ...] = ("method", "small_molecule", "gene", "protein")
    """Kinds that imply a handle to intervene on or measure."""


class RankingParams(BaseModel):
    """The model stages: articulate, attack, compare, evolve.

    This follows the co-scientist generate/debate/evolve loop, with one
    change: nothing here may introduce a fact. Articulation is pinned to the
    link and finding ids the deterministic stages selected, and a critique that
    cites anything else is rejected and the candidate reopened.
    """

    model_config = ConfigDict(extra="forbid")

    articulate: bool = True
    critique: bool = True
    max_claims_per_hypothesis: int = 6

    critic_lenses: tuple[str, ...] = ("mechanism", "evidence", "testability")
    """Each critic gets one lens. Diverse lenses catch failure modes that N
    identical refuters cannot; identical critics mostly agree with each other.

    This is also what stands in for sampling temperature. There is deliberately
    no ``temperature`` or ``seed`` knob here: Claude Opus 5 rejects
    ``temperature``/``top_p``/``top_k`` with a 400, so a "hot to write, cold to
    judge" dial would be a field that quietly does nothing. Depth is controlled
    by ``output_config.effort`` in ``llm.py``, diversity by these lenses, and
    reproducibility by deterministic tie-breaks in ``select.py`` -- all three
    of which work better than resampling the same prompt anyway."""

    critics_per_hypothesis: int = 2
    refute_threshold: float = 0.5
    """Fraction of critics that must call a hypothesis unsupported before it is
    dropped rather than flagged."""

    tournament: bool = False
    """Elo tournament with pairwise debates, as in the co-scientist. Off by
    default because it is O(k log k) model calls for a ranking the scores
    already approximate -- turn it on when the top few are genuinely close."""

    elo_initial: float = 1200.0
    elo_k: float = 32.0
    tournament_matches_per_hypothesis: int = 4
    debate_turns: int = 2
    """Judging passes per pair, alternating which hypothesis is presented
    first. Pairwise judges have a position bias, so 1 buys a ranking that
    partly reflects presentation order; 2 makes that bias show up as a split
    verdict, which moves Elo by almost nothing. Odd values just cost a call."""

    evolution_rounds: int = 0
    """Rounds of refine-and-resubmit over the current leaders. 0 keeps the run
    single-pass, which is the honest MVP setting."""

    evolve_top_n: int = 3
    evolve_operators: tuple[str, ...] = ("specialise", "combine", "invert_condition")

    effort_articulate: str = "high"
    effort_critique: str = "high"
    """Thinking depth per stage: low | medium | high | xhigh | max.

    This is the replacement for a temperature dial, which this model does not
    accept (see ``critic_lenses``). Effort is the real cost/quality control --
    sweep both down to `medium` against a retrospective set before assuming the
    default is needed."""


class VerificationParams(BaseModel):
    """The staged gate process that runs over each articulated hypothesis.

    Gates run in the order listed, and the order is by cost: every deterministic
    check that could kill a hypothesis runs before the one gate that spends
    model calls. Reordering so `adversarial` comes earlier is legal and will
    work, it just pays for critics on hypotheses a free check was about to
    reject.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    gates: tuple[str, ...] = (
        "structure",
        "citations",
        "consistency",
        "independence",
        "falsifiability",
        "adversarial",
    )
    """Which gates run, and in what order. Drop a name to skip that gate
    entirely -- it will not appear in the table, rather than appearing as a
    skip, because "we chose not to check" and "we could not check" are
    different things and the table should not blur them."""

    halt_on: tuple[str, ...] = ("structure", "citations", "independence")
    """Gates whose failure stops the process. Everything downstream is recorded
    as a skip naming the halting gate, so a report never implies a check passed
    when it simply never ran.

    `structure` and `citations` halt because they mean the output is not
    trustworthy. `independence` halts because a hypothesis resting on one
    research group is not a thing adversarial critics can fix -- and note it
    only *fails* when the run's own ``evidence.min_independent_groups`` is
    unmet, so under the default profile (which asks for 1) it warns instead."""

    require_primary_evidence: bool = True
    """Fail `independence` when every finding on the path is a citation of
    someone else's result. Secondhand support is not support; the graph read a
    paper that read a paper."""

    max_claim_overlap: float = 0.9
    """Token overlap above which the falsifier is treated as a restatement of
    the statement rather than a falsifier. "This hypothesis is false" is not an
    observation that would kill anything."""

    min_falsifier_chars: int = 25
    """Below this, a falsifier is warned about as too terse to act on."""


class LoopParams(BaseModel):
    """When to stop reasoning and ask the graph builder for more graph.

    The generator is blind to the graph builder's machinery but can name a row by id, and
    that is enough to close the loop. Each trigger below maps to exactly one
    `ask`, because the contract permits one ask per request.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_requests: int = 3
    depth: Literal["quick", "standard", "deep", "exhaustive"] = "standard"

    test_gap_when_ranked_above: int = 3
    """`test_gap` a gap that is driving a top-N hypothesis and has
    `searched_in_round: null`. The whole hypothesis rests on nobody having
    looked; find out whether anybody did."""

    resolve_link_below_confidence: float = 0.35
    """`resolve_link` on any link a surviving hypothesis depends on whose state
    is `disagreed` or `single_source` and whose confidence is this low."""

    expand_node_min_mentions: int = 8
    expand_node_max_degree: int = 2
    """`expand_node` on a thing the literature talks about constantly but that
    the graph has barely connected -- the signature of an under-read node."""

    stop_when_no_score_change: float = 0.02
    """Stop the loop when a full round moves no surviving score by more than
    this. More graph that changes nothing is a stopping condition, not a
    failure."""


class BudgetParams(BaseModel):
    """Ceilings. Every one of these is a hard stop, not a target."""

    model_config = ConfigDict(extra="forbid")

    max_model_calls: int = 40
    max_enumeration_seconds: float = 30.0
    max_output_hypotheses: int = 12


class StanceParams(BaseModel):
    """How this Params was derived. A record, not a knob.

    Nothing in the pipeline reads these to make a decision -- they exist so a
    record can say where its numbers came from, and so ``at_craziness`` is
    idempotent. That distinction matters: params.py's objection to a dial like
    ``temperature`` is that it would be a field that quietly does nothing, and
    the answer to that is not to hide provenance but to label it as provenance.
    """

    model_config = ConfigDict(extra="forbid")

    profile: str = "default"
    craziness: float | None = None


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stance: StanceParams = Field(default_factory=StanceParams)
    framing: FramingParams = Field(default_factory=FramingParams)
    traversal: TraversalParams = Field(default_factory=TraversalParams)
    motifs: MotifParams = Field(default_factory=MotifParams)
    evidence: EvidenceParams = Field(default_factory=EvidenceParams)
    novelty: NoveltyParams = Field(default_factory=NoveltyParams)
    selection: SelectionParams = Field(default_factory=SelectionParams)
    ranking: RankingParams = Field(default_factory=RankingParams)
    verification: VerificationParams = Field(default_factory=VerificationParams)
    loop: LoopParams = Field(default_factory=LoopParams)
    budget: BudgetParams = Field(default_factory=BudgetParams)

    @classmethod
    def load(cls, path: str | Path | None) -> "Params":
        if path is None:
            return cls()
        return cls.model_validate(json.loads(Path(path).read_text()))

    @classmethod
    def profile(cls, name: str, overrides: dict | None = None) -> "Params":
        """A named profile, optionally patched. Overrides are a nested dict so
        a CLI can pass ``{"traversal": {"max_hops": 4}}`` without restating the
        rest of the profile."""
        base = PROFILES[name].model_dump()
        base["stance"] = {"profile": name, "craziness": base["stance"]["craziness"]}
        for group, values in (overrides or {}).items():
            base.setdefault(group, {}).update(values)
        return cls.model_validate(base)

    @classmethod
    def at_craziness(
        cls,
        craziness: float,
        base: str = "default",
        overrides: dict | None = None,
    ) -> "Params":
        """One dial from super-safe (0.0) to very ambitious (1.0).

        A profile says *what question* to ask the graph; craziness says *how far
        out* to reach for an answer. They compose: ``repurposing`` at 0.2 and
        ``repurposing`` at 0.9 ask the same question of the same shape and come
        back with very different records.

        The scale is not invented. ``conservative``, ``default`` and
        ``speculative`` were already three points on exactly this axis, so
        craziness makes them continuous rather than replacing them: 0.0 and 0.5
        reproduce the first two, everything between is piecewise-linear, and the
        three profiles remain as names for the places people actually stop.

        1.0 deliberately differs from ``speculative`` in two ways. It reaches one
        hop further, and it does **not** inherit that profile's
        ``min_novelty=0.4`` -- a floor which, because novelty is measured as
        distance, silently excludes every analogical transfer and hands back a
        record of long chains. See ``CRAZINESS_NEVER_TOUCHES``. The profile is
        left as it is; the dial does not copy the bug.

        Precedence is profile → craziness → ``overrides``, last wins, so
        ``--set traversal.max_hops=2`` still pins a knob at any craziness.
        """

        if not 0.0 <= craziness <= 1.0:
            raise ValueError(f"craziness must be between 0 and 1, got {craziness}")

        values = PROFILES[base].model_dump()
        for path, anchors in _CRAZINESS_SCHEDULE.items():
            _write(values, path, _interpolate(anchors, craziness))
        _apply_craziness_cliffs(values, craziness)
        values["stance"] = {"profile": base, "craziness": craziness}

        for group, patch in (overrides or {}).items():
            values.setdefault(group, {}).update(patch)
        return cls.model_validate(values)


# -- the craziness dial ----------------------------------------------------
#
# Each row is (at 0.0, at 0.5, at 1.0), piecewise-linear between them, cast back
# to the field's type. Three anchors rather than two because one knob is
# genuinely not monotonic -- see hub_damping.

_CRAZINESS_SCHEDULE: dict[str, tuple[float, float, float]] = {
    # The speculation dial, and the first thing to reach for. Every extra hop
    # multiplies the ways the story can be wrong, which is the whole trade.
    "traversal.max_hops": (2, 3, 5),
    # How weak a link may be before traversal refuses to walk it. A crazy run is
    # willing to build on a shakier stated relationship; it does not get to
    # pretend the relationship is stronger than it is (see `scoring`, below).
    "traversal.min_link_confidence": (0.45, 0.20, 0.05),
    # NOT monotonic, and this is the point. Damping *rises* again at high
    # craziness because the extra hops are only worth having if they are not all
    # routed through one promiscuous node -- reaching further and reaching
    # through a hub are different things, and only the first is ambition.
    "traversal.hub_damping": (0.60, 0.40, 0.55),
    "traversal.max_paths_per_pair": (1, 2, 3),
    "traversal.max_branch_per_node": (8, 12, 20),
    # Crossing a link against its stated direction is a weaker claim, not an
    # invalid one, so it is an ambition knob -- with the penalty softening as
    # craziness rises rather than vanishing.
    "traversal.reversal_penalty": (0.70, 0.70, 0.85),
    # The "I read this in a slightly different field" knobs. Lower floors mean a
    # thinner resemblance is enough to propose a transfer.
    "motifs.analogy_min_shared": (3, 2, 1),
    "motifs.analogy_min_jaccard": (0.30, 0.15, 0.05),
    # Analogical transfer is the motif that reasons from similarity rather than
    # from a path: the most likely to be fluent and wrong, and the one that
    # carries the ambition. It leads the record only at the top of the dial.
    "motifs.weights.analogical_transfer": (0.35, 0.70, 1.00),
    "motifs.weights.transitive_chain": (0.85, 0.90, 0.95),
    "motifs.weights.gap_closure": (1.00, 1.00, 0.90),
    "motifs.weights.condition_split": (0.90, 0.85, 0.80),
    # Novelty that rests on distance, and the correction for the fact that a
    # novelty judge over-rewards famous, densely-connected pairings. A safe run
    # barely penalises popularity because it *wants* the well-trodden answer.
    "novelty.hop_novelty": (0.15, 0.22, 0.30),
    "novelty.popularity_penalty": (0.05, 0.15, 0.35),
    # A safe run demands support; an ambitious one stops demanding it. Note what
    # is *not* here: `min_novelty`. See CRAZINESS_NEVER_TOUCHES -- raising it is
    # the one obvious-looking move on this dial that is actively wrong.
    "selection.min_support": (0.40, 0.00, 0.00),
    "selection.top_k": (5, 8, 12),
    "selection.diversity_lambda": (0.85, 0.70, 0.50),
    # Scrutiny rises with craziness, it does not fall. A speculative record's
    # failure mode is fluent nonsense, so the ambitious end of the dial buys
    # more critics and a revision round -- the opposite of relaxing.
    "ranking.critics_per_hypothesis": (2, 2, 3),
    "ranking.evolution_rounds": (0, 0, 1),
}

_INTEGER_FIELDS = frozenset(
    {
        "traversal.max_hops",
        "traversal.max_paths_per_pair",
        "traversal.max_branch_per_node",
        "motifs.analogy_min_shared",
        "selection.top_k",
        "ranking.critics_per_hypothesis",
        "ranking.evolution_rounds",
    }
)


def _interpolate(anchors: tuple[float, float, float], craziness: float) -> float:
    low, middle, high = anchors
    if craziness <= 0.5:
        return low + (middle - low) * (craziness / 0.5)
    return middle + (high - middle) * ((craziness - 0.5) / 0.5)


def _write(values: dict, path: str, value: float) -> None:
    *parents, leaf = path.split(".")
    scope = values
    for name in parents:
        scope = scope[name]
    scope[leaf] = round(value) if path in _INTEGER_FIELDS else round(value, 4)


def _apply_craziness_cliffs(values: dict, craziness: float) -> None:
    """The knobs that step rather than slide.

    A boolean cannot be 0.4 true, and pretending otherwise by thresholding a
    lerp would hide where the step actually is. These are written out so the
    three places the dial changes character are visible and testable.
    """

    # Below this, a chain may not be read backwards at all -- CONSERVATIVE's
    # stance, for strictly causal reading.
    values["traversal"]["allow_edge_reversal"] = craziness >= 0.25

    # One lab reporting a result five times is one result. A safe run refuses to
    # rest on that; past the bottom of the dial it becomes a warning instead.
    values["evidence"]["min_independent_groups"] = 2 if craziness < 0.25 else 1
    halting = list(values["verification"]["halt_on"])
    if craziness >= 0.25 and "independence" in halting:
        halting.remove("independence")
    values["verification"]["halt_on"] = tuple(halting)

    # Analogy across kinds is usually a category error -- a small molecule is
    # not like a disease. At the very top of the dial that guard comes off,
    # which is the most literal reading of "it worked in a different field".
    values["motifs"]["analogy_same_kind_only"] = craziness < 0.75

    # At the bottom, the similarity motif does not run at all rather than
    # running at a low weight. Craziness may only ever *narrow* what the base
    # profile enabled, never add a motif the profile deliberately excluded.
    if craziness < 0.20:
        values["motifs"]["enabled"] = tuple(
            m for m in values["motifs"]["enabled"] if m != "analogical_transfer"
        )


CRAZINESS_NEVER_TOUCHES: frozenset[str] = frozenset(
    {
        # -- the evidence arithmetic ------------------------------------
        # Craziness changes what you are willing to *propose*. It must never
        # change what the evidence *says*. The same chain of links scores the
        # same support at 0.1 and at 0.9; if it did not, the dial would be a
        # licence to launder a weak chain into a strong one.
        #
        # `min_independent_groups` is the deliberate exception and is not listed
        # here. It is a *standard* rather than a weight -- how many labs this run
        # requires before a link may exceed `single_group_cap` -- and requiring
        # less corroboration is precisely what a more ambitious run is doing. It
        # says so out loud when it bites ("support capped: a link rests on a
        # single research group"), which is the condition for letting it move.
        "evidence.study_weights",
        "evidence.hedged_penalty",
        "evidence.secondhand_penalty",
        "evidence.preprint_penalty",
        "evidence.basis_penalty",
        "evidence.single_group_cap",
        "evidence.support_weights",
        "evidence.chain_aggregation",
        # -- novelty is a length measure, not an ambition measure -------
        # The tempting move is to raise `min_novelty` with craziness. It is
        # wrong, and measurably so. Novelty here is *distance from what is
        # already stated* -- hops beyond the first, plus gap bonuses -- so an
        # analogical transfer, which is a single bridge edge, scores low on it
        # by construction no matter how audacious the leap. A novelty floor is
        # therefore a path-length filter wearing a novelty label: on the demo
        # graph, `min_novelty=0.4` removes every one of the 90 enumerated
        # analogical transfers and returns a record of twelve long chains.
        #
        # That is exactly backwards. "I read this in a slightly different field,
        # maybe it works here" is the most ambitious thing this generator can
        # say, and it is short. Ambition belongs in the aperture -- hops,
        # confidence floors, the Jaccard floor, cross-kind analogy, motif
        # weights -- not in a filter that only long paths can clear.
        "selection.min_novelty",
        # -- absence is still not evidence of absence -------------------
        # A shallow graph may not mint novelty at any ambition level. Turning
        # this off would let craziness manufacture the very thing it is most
        # tempted to claim: that nobody has looked.
        "novelty.respect_absence_reliability",
        "novelty.gap_confidence_cap",
        # -- the audit standard -----------------------------------------
        # An ambitious hypothesis is still not allowed to cite what it was not
        # shown, restate a fact the graph already contains, or come back from a
        # broken path. Craziness widens the aperture; it never lowers the bar
        # for what counts as a checkable claim.
        "motifs.require_unstated",
        "verification.enabled",
        "verification.require_primary_evidence",
        "verification.max_claim_overlap",
        # -- inference forms that are wrong, not bold -------------------
        # Two correlations in a row imply nothing, and "A does not do B, B does
        # C" composes to nothing. Chaining them is not ambition, it is a broken
        # inference wearing ambition's coat. Flip them by hand with --set if you
        # want them; the dial will not do it for you.
        "traversal.predicates_deny",
        "traversal.allow_no_effect_edges",
        "traversal.allow_negative_edges",
    }
)
"""Fields the dial is forbidden to move, and why.

`structure` and `citations` also stay in ``verification.halt_on`` at every
level: they mean the output is untrustworthy, which is orthogonal to how
ambitious it was trying to be. Only `independence` moves, because "one lab" is
a statement about how much corroboration you require -- which is exactly what
this dial is for.
"""


CONSERVATIVE = Params(
    traversal=TraversalParams(
        max_hops=2,
        min_link_confidence=0.45,
        allow_edge_reversal=False,
        hub_damping=0.6,
    ),
    motifs=MotifParams(enabled=("gap_closure", "condition_split")),
    evidence=EvidenceParams(min_independent_groups=2),
    novelty=NoveltyParams(popularity_penalty=0.05),
    selection=SelectionParams(top_k=5, min_support=0.4, diversity_lambda=0.85),
)
"""For a clinical audience. Short paths, strong links, two labs, no reversals.
Produces fewer hypotheses, and the ones it produces are nearly boring -- which
is the point when the next step costs money."""

SPECULATIVE = Params(
    traversal=TraversalParams(
        max_hops=4,
        min_link_confidence=0.10,
        max_paths_per_pair=3,
        hub_damping=0.5,
    ),
    novelty=NoveltyParams(popularity_penalty=0.30, hop_novelty=0.28),
    selection=SelectionParams(top_k=12, min_novelty=0.4, diversity_lambda=0.5),
    ranking=RankingParams(critics_per_hypothesis=3, evolution_rounds=1),
)
"""For an exploratory audience. Longer paths and weaker links, with the hub
damping raised to compensate -- the extra hops are only worth having if they
are not all routed through one promiscuous node -- and more critics, because
this profile's failure mode is fluent nonsense."""

REPURPOSING = Params(
    traversal=TraversalParams(
        max_hops=4,
        seed_kinds=("small_molecule",),
        target_kinds=("disease",),
        intermediate_kinds=("protein", "gene", "process"),
        hub_damping=0.4,
    ),
    motifs=MotifParams(enabled=("transitive_chain", "analogical_transfer", "gap_closure")),
    selection=SelectionParams(top_k=10, max_per_object=3),
)
"""The Rephetio shape: compound -> gene/protein -> process -> disease. Kind
constraints do most of the work; set ``traversal.metapaths`` instead if you
want the exact sequences rather than the loose shape."""

MECHANISM = Params(
    framing=FramingParams(mode="closed"),
    traversal=TraversalParams(max_hops=3, min_link_confidence=0.3),
    motifs=MotifParams(enabled=("transitive_chain", "condition_split")),
    selection=SelectionParams(top_k=8, similarity="endpoint", diversity_lambda=0.6),
)
"""Closed discovery: both ends are given, the output is the B terms that
explain them. Run this when the clinical prompt already names a drug and a
disease and the question is *why*."""

VALUATION = Params(
    traversal=TraversalParams(
        max_hops=3,
        seed_kinds=("small_molecule",),
        target_kinds=("disease",),
        intermediate_kinds=("protein", "gene", "process"),
        hub_damping=0.4,
    ),
    motifs=MotifParams(
        enabled=("transitive_chain", "gap_closure", "analogical_transfer"),
        weights={
            "gap_closure": 1.00,
            "transitive_chain": 1.00,
            "analogical_transfer": 0.50,
            "condition_split": 0.85,
        },
    ),
    evidence=EvidenceParams(min_independent_groups=2),
    selection=SelectionParams(
        top_k=6,
        min_support=0.35,
        max_per_subject=2,
        max_per_object=2,
        diversity_lambda=0.75,
    ),
)
"""Shaped by what the valuation stage can actually evaluate.

`repurposing` asks the graph a scientific question; this profile asks it a
question LABrador can price, and every difference between the two is a downstream
constraint rather than a taste:

- **Intervention in, disease out.** LABrador values an asset against an
  indication, so `seed_kinds`/`target_kinds` are not optional here the way they
  are in an exploratory run. A hypothesis ending on a process is a fine
  hypothesis and not a program; `valuation.py` skips it by name.
- **A protein or gene in the middle.** `ProgramInput.target` is read off the
  first mechanism node the path crosses. Chains are therefore weighted level with
  gaps -- they are the motif that reliably supplies one -- and
  `intermediate_kinds` keeps the middle of the path somewhere a target can be
  found.
- **Two labels per molecule, no more.** `max_per_subject=2` is LABrador's
  two-indication cash-flow model showing through: one asset, one patent clock,
  an initial indication and at most one expansion. A third label on the same
  molecule cannot be valued and is reported as dropped rather than emitted.
- **Three hops, not four.** Every hop is another claim an analyst has to defend
  in a brief that ends in a number, and testability already penalises length.
- **Two independent groups.** LABrador clears a critical input only on
  HIGH/MODERATE non-synthetic evidence. A record resting on one lab produces
  programs that cannot clear anything, so the honest place to spend that
  constraint is here, before the model calls -- not downstream as a surprise.
- **`analogical_transfer` halved rather than banned.** It is genuinely useful and
  structurally weak for this purpose: its path is the *donor's* bridge edge, so
  it yields no mechanism node and its program is emitted with an `UNSPECIFIED`
  target and the donor caveat attached. Worth seeing, worth ranking below a chain.

What this profile deliberately does not do is let the downstream number touch the
science. Nothing here scores a hypothesis by how valuable the program would be:
market size is not evidence, and a generator that preferred lucrative hypotheses
would be optimising the one axis its own evidence cannot check.
"""

PROFILES: dict[str, Params] = {
    "default": Params(),
    "conservative": CONSERVATIVE,
    "speculative": SPECULATIVE,
    "repurposing": REPURPOSING,
    "mechanism": MECHANISM,
    "valuation": VALUATION,
}
