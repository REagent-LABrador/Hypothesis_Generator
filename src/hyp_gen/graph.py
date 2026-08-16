"""The knowledge graph, as received, plus the indexes traversal needs.

This module is the whole world. The generator is blind to where the graph came
from -- no PubMed, no search strategy, no question text beyond the string in
``question``. Everything downstream reads the graph through ``GraphIndex``, so
if a fact is not reachable here it cannot appear in a hypothesis.

Models are permissive on unknown fields (``extra="allow"``): the graph builder owns the
schema and will grow it, and a new key should never crash a run. They are
strict on the fields the traversal actually depends on.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # graph.py stays importable without the knobs
    from hyp_gen.params import TraversalParams

Says = Literal["yes", "no", "no_effect"]
LinkState = Literal["agreed", "disagreed", "single_source", "no_effect"]
Basis = Literal["primary", "hedged_only", "background_only", "mixed"]
Depth = Literal["quick", "standard", "deep", "exhaustive"]
Kind = Literal[
    "protein", "small_molecule", "gene", "disease", "process", "method"
]

# Ordering matters: `quick` reads page one, and page one lies. Absence of a
# link at `quick` means "unknown", not "nobody found it" -- see notes 2 in the
# schema. Anything that reasons from absence has to know which tier it is on.
DEPTH_RANK: dict[str, int] = {
    "quick": 0,
    "standard": 1,
    "deep": 2,
    "exhaustive": 3,
}

STUDY_TYPES = (
    "meta_analysis",
    "clinical_trial",
    "human_cohort",
    "animal",
    "test_tube",
    "computational",
    "review",
)


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Thing(_Base):
    id: str
    name: str
    kind: str = "process"
    aliases: list[str] = Field(default_factory=list)
    mentions: int = 0


class Paper(_Base):
    id: str
    title: str = ""
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    first_author: str | None = None
    study_type: str = "computational"
    is_preprint: bool = False
    round: int = 1


class Finding(_Base):
    id: str
    src: str = Field(alias="from")
    how: str
    dst: str = Field(alias="to")
    says: str = "yes"
    quote: str = ""
    paper: str
    where: str | None = None
    is_own_result: bool = True
    hedged: bool = False
    confidence: float = 0.5
    flags: list[str] = Field(default_factory=list)
    round: int = 1


class LinkConfidence(_Base):
    overall: float = 0.0
    label: str = "low"
    evidence_quality: float = 0.0
    agreement: float = 0.0
    independence: float = 0.0


class Link(_Base):
    id: str
    src: str = Field(alias="from")
    how: str
    dst: str = Field(alias="to")
    yes: list[str] = Field(default_factory=list)
    no: list[str] = Field(default_factory=list)
    no_effect: list[str] = Field(default_factory=list)
    state: str = "single_source"
    why: str | None = None
    basis: str = "primary"
    confidence: LinkConfidence = Field(default_factory=LinkConfidence)
    changed_in_round: int | None = None

    @property
    def finding_ids(self) -> list[str]:
        return [*self.yes, *self.no, *self.no_effect]


class Gap(_Base):
    id: str
    missing: list[str]
    implied_by: list[str] = Field(default_factory=list)
    note: str = ""
    confidence: float = 0.0
    searched_in_round: int | None = None


class Limits(_Base):
    max_papers: int | None = None
    max_queries: int | None = None
    hit_limit: str | None = None


class Coverage(_Base):
    depth: str = "standard"
    found: int = 0
    read: int = 0
    used: int = 0
    truncated: bool = False
    no_quote_discarded: int = 0
    limits: Limits = Field(default_factory=Limits)


class Round(_Base):
    n: int
    ask: str
    target: str | None = None
    depth: str = "standard"
    papers_added: int = 0


class KnowledgeGraph(_Base):
    schema_version: str = "1.1"
    graph_id: str
    question: str = ""
    round: int = 1
    generated_at: str | None = None
    rounds: list[Round] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    things: list[Thing] = Field(default_factory=list)
    papers: list[Paper] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        return cls.model_validate(json.loads(Path(path).read_text()))


class Edge(BaseModel):
    """One traversal step: a link, plus the direction it was crossed in.

    Direction is kept because ``binds`` and ``inhibits`` are not symmetric, and
    a chain that silently reverses an arrow is a different (usually wrong)
    claim. ``forward=False`` means the hypothesis is reading the link
    backwards, which the articulation prompt is told about explicitly.
    """

    model_config = ConfigDict(frozen=True)

    link_id: str
    src: str
    dst: str
    how: str
    forward: bool


class GraphIndex:
    """Read-only lookups and traversal over one graph at one round.

    Nothing here caches across graphs or rounds. Link confidences move between
    rounds by design (schema note 4), so an index is valid for exactly one
    ``(graph_id, round)`` pair and is rebuilt whenever the graph is reloaded.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph
        self.things: dict[str, Thing] = {t.id: t for t in graph.things}
        self.papers: dict[str, Paper] = {p.id: p for p in graph.papers}
        self.findings: dict[str, Finding] = {f.id: f for f in graph.findings}
        self.links: dict[str, Link] = {l.id: l for l in graph.links}
        self.gaps: dict[str, Gap] = {g.id: g for g in graph.gaps}

        self._adj: dict[str, list[Edge]] = defaultdict(list)
        self._between: dict[frozenset[str], list[Link]] = defaultdict(list)
        for link in graph.links:
            if link.src not in self.things or link.dst not in self.things:
                continue  # dangling endpoint: not traversable, not our bug
            self._adj[link.src].append(
                Edge(link_id=link.id, src=link.src, dst=link.dst, how=link.how, forward=True)
            )
            self._adj[link.dst].append(
                Edge(link_id=link.id, src=link.dst, dst=link.src, how=link.how, forward=False)
            )
            self._between[frozenset((link.src, link.dst))].append(link)

        self._gap_by_pair: dict[frozenset[str], Gap] = {}
        for gap in graph.gaps:
            if len(gap.missing) == 2:
                self._gap_by_pair[frozenset(gap.missing)] = gap

    # -- basic lookups ----------------------------------------------------

    def name(self, thing_id: str) -> str:
        thing = self.things.get(thing_id)
        return thing.name if thing else thing_id

    def kind(self, thing_id: str) -> str:
        thing = self.things.get(thing_id)
        return thing.kind if thing else "process"

    def degree(self, thing_id: str) -> int:
        return len(self._adj.get(thing_id, ()))

    def neighbors(self, thing_id: str) -> list[Edge]:
        return list(self._adj.get(thing_id, ()))

    def neighbor_ids(self, thing_id: str) -> set[str]:
        return {e.dst for e in self._adj.get(thing_id, ())}

    def links_between(self, a: str, b: str) -> list[Link]:
        return list(self._between.get(frozenset((a, b)), ()))

    def gap_between(self, a: str, b: str) -> Gap | None:
        return self._gap_by_pair.get(frozenset((a, b)))

    def findings_for(self, link: Link) -> list[Finding]:
        return [self.findings[f] for f in link.finding_ids if f in self.findings]

    def paper_for(self, finding: Finding) -> Paper | None:
        return self.papers.get(finding.paper)

    def papers_for(self, link: Link) -> list[Paper]:
        seen: dict[str, Paper] = {}
        for finding in self.findings_for(link):
            paper = self.paper_for(finding)
            if paper is not None:
                seen[paper.id] = paper
        return list(seen.values())

    def conditions_for(self, link: Link) -> list[str]:
        """Distinct experimental conditions the findings were observed under.

        Schema note 5: a `disagreed` link usually is not a conflict -- it is
        two conditions. This is what makes that check possible downstream.
        """
        out: list[str] = []
        for finding in self.findings_for(link):
            where = (finding.where or "").strip()
            if where and where not in out:
                out.append(where)
        return out

    def is_negative(self, link: Link) -> bool:
        """The findings on this link mostly say the relationship is not there."""
        return len(link.no) > len(link.yes)

    # -- traversal --------------------------------------------------------

    def edge_ok(self, edge: Edge, params: "TraversalParams") -> bool:
        """Every per-edge gate, in one place.

        Ordered cheapest-first, and deliberately *not* including anything
        path-dependent -- hub damping and metapath matching judge the whole
        path and live in ``dwpc`` and ``walk``.
        """
        link = self.links[edge.link_id]
        if link.confidence.overall < params.min_link_confidence:
            return False
        if not params.allow_no_effect_edges and link.state == "no_effect":
            return False
        if not params.allow_negative_edges and self.is_negative(link):
            return False
        if not params.allow_edge_reversal and not edge.forward:
            return False
        if params.predicates_deny and link.how in params.predicates_deny:
            return False
        if params.predicates_allow and link.how not in params.predicates_allow:
            return False
        if params.max_node_degree and self.degree(edge.dst) > params.max_node_degree:
            return False
        return True

    def dwpc(self, path: Iterable[Edge], damping: float) -> float:
        """Degree-weighted path count for a single path.

        Rephetio's device for the fact that a path is not evidence in
        proportion to its existence. Each edge is divided by
        ``(deg(src) * deg(dst)) ** w``, so a hop through a node that touches
        everything contributes almost nothing, while a hop between two
        sparsely-connected things contributes nearly its full weight. This is
        what stops "aspirin -> inflammation -> everything" from being the top
        hypothesis on every graph.

        Reversed hops are additionally scaled by ``reversal_penalty`` at the
        call site rather than here, so this stays a pure structural quantity.
        """
        total = 1.0
        for edge in path:
            d = self.degree(edge.src) * self.degree(edge.dst)
            if d <= 0:
                return 0.0
            total *= 1.0 / (d ** damping)
        return total

    def path_weight(self, path: Iterable[Edge], params: "TraversalParams") -> float:
        """DWPC, discounted for every hop crossed against its stated arrow."""
        edges = tuple(path)
        weight = self.dwpc(edges, params.hub_damping)
        for edge in edges:
            if not edge.forward:
                weight *= params.reversal_penalty
        return weight

    def _kinds_ok(self, start: str, path: list[Edge], params: "TraversalParams") -> bool:
        """Metapath match if one is declared, loose kind filters otherwise."""
        sequence = [self.kind(start), *(self.kind(e.dst) for e in path)]
        if params.metapaths:
            return any(tuple(sequence) == tuple(mp) for mp in params.metapaths)
        if params.intermediate_kinds:
            if any(k not in params.intermediate_kinds for k in sequence[1:-1]):
                return False
        if params.target_kinds and sequence[-1] not in params.target_kinds:
            return False
        return True

    def _prefix_possible(self, start: str, path: list[Edge], params: "TraversalParams") -> bool:
        """Can this partial path still become a metapath match? Prunes the BFS
        instead of enumerating every path and filtering at the end."""
        if not params.metapaths:
            return True
        sequence = tuple([self.kind(start), *(self.kind(e.dst) for e in path)])
        return any(
            len(mp) >= len(sequence) and tuple(mp[: len(sequence)]) == sequence
            for mp in params.metapaths
        )

    def walk(
        self, start: str, params: "TraversalParams"
    ) -> Iterator[tuple[list[Edge], float]]:
        """Enumerate simple paths out of ``start``, shortest first, with weights.

        Breadth-first so that the shortest connection between two things is
        found before longer ones -- a 4-hop story about a pair that is already
        2 hops apart is noise, not a hypothesis. Paths are capped per target
        rather than globally so one dense hub cannot starve the rest of the
        graph, and the frontier is capped per node so a hub cannot starve the
        frontier either.

        Yields ``(path, weight)``; the weight is the reversal-discounted DWPC,
        already filtered against ``min_dwpc``.
        """
        emitted: dict[str, int] = defaultdict(int)
        frontier: list[list[Edge]] = [[]]
        for _ in range(params.max_hops):
            nxt: list[list[Edge]] = []
            for path in frontier:
                node = path[-1].dst if path else start
                visited = {start, *(e.dst for e in path)}
                candidates = [
                    e
                    for e in self._adj.get(node, ())
                    if e.dst not in visited and self.edge_ok(e, params)
                ]
                # Beam: keep the most confident branches when a node is dense.
                if params.max_branch_per_node:
                    candidates.sort(
                        key=lambda e: self.links[e.link_id].confidence.overall,
                        reverse=True,
                    )
                    candidates = candidates[: params.max_branch_per_node]
                for edge in candidates:
                    extended = [*path, edge]
                    if not self._prefix_possible(start, extended, params):
                        continue
                    weight = self.path_weight(extended, params)
                    if (
                        emitted[edge.dst] < params.max_paths_per_pair
                        and weight >= params.min_dwpc
                        and self._kinds_ok(start, extended, params)
                    ):
                        emitted[edge.dst] += 1
                        yield extended, weight
                    nxt.append(extended)
            frontier = nxt
            if not frontier:
                return

    def paths(
        self,
        start: str,
        params: "TraversalParams",
    ) -> Iterator[list[Edge]]:
        """``walk`` without the weights, for callers that only want the shape."""
        for path, _ in self.walk(start, params):
            yield path

    # -- coverage-derived caveats ----------------------------------------

    @property
    def depth_rank(self) -> int:
        return DEPTH_RANK.get(self.graph.coverage.depth, 1)

    def absence_reliability(self) -> float:
        """How much a *missing* link is allowed to count as evidence of absence.

        A hypothesis built on "nobody has connected these" is only interesting
        if somebody looked. At `quick` depth nobody did, so absence carries no
        information at all; a truncated search is a sample of the literature,
        not the literature. This factor gates every novelty score that leans on
        a gap, which is the difference between "unexplored" and "unread".
        """
        base = [0.0, 0.45, 0.8, 1.0][self.depth_rank]
        if self.graph.coverage.truncated:
            base *= 0.6
        if self.graph.coverage.limits.hit_limit:
            base *= 0.85
        return round(base, 3)
