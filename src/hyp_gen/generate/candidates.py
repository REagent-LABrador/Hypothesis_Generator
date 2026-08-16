"""Structural hypothesis enumeration.

A hypothesis starts as a *shape* in the graph, not as a sentence. Finding the
shape deterministically and only then handing it to a model is what keeps the
output auditable: every candidate below carries the exact link and finding ids
it came from, so a reader can walk back from a claim to a quote without
trusting the model's memory.

Four motifs, each a different reason a statement is worth making:

``gap_closure``
    The graph builder noticed a pair that its own links imply but nobody states. The
    hypothesis is that the implied relation is real.

``transitive_chain``
    A -> B -> C exists, A -> C does not. The hypothesis composes the chain.
    Sub-tagged ``repurposing`` when it lands an intervention-shaped thing on a
    disease, which is the shape the ROI stage cares about.

``analogical_transfer``
    X and Y share several neighbours, X has an edge Y lacks. The hypothesis is
    that Y has it too. This is the only motif that reasons from similarity
    rather than from a path, and it is the most speculative.

``condition_split``
    A link is `disagreed`, and its findings were observed under different
    ``where`` conditions. The hypothesis is that both results are right and the
    condition is the variable. Schema note 5 says this is the common case, so
    it is treated as a first-class hypothesis rather than a data problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from hyp_gen.graph import Edge, GraphIndex
from hyp_gen.params import Params

INTERVENTION_KINDS = {"small_molecule", "protein", "gene", "method"}


@dataclass(frozen=True)
class Candidate:
    """One structural hypothesis, before any language is put on it."""

    id: str
    motif: str
    subject: str
    object: str
    path: tuple[Edge, ...] = ()
    gap_id: str | None = None
    focus_link_id: str | None = None
    analogues: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    note: str = ""
    weight: float = 1.0
    """Reversal-discounted DWPC of ``path``. Carried from enumeration rather
    than recomputed later, because it is the only score that depends on the
    traversal that found it."""

    @property
    def link_ids(self) -> tuple[str, ...]:
        ids = [e.link_id for e in self.path]
        if self.focus_link_id and self.focus_link_id not in ids:
            ids.append(self.focus_link_id)
        return tuple(ids)

    @property
    def hops(self) -> int:
        return max(len(self.path), 1)

    def node_ids(self) -> tuple[str, ...]:
        seen = [self.subject]
        for edge in self.path:
            if edge.dst not in seen:
                seen.append(edge.dst)
        if self.object not in seen:
            seen.append(self.object)
        return tuple([*seen, *(a for a in self.analogues if a not in seen)])

    def key(self) -> tuple:
        """Dedupe key. Endpoints *and* links are unordered: A->B and B->A over
        the same links are one hypothesis stated from two ends, and BFS from
        every seed will find both. Ordering the link ids here is what stops the
        record from being half mirror images."""
        return (self.motif, frozenset((self.subject, self.object)), frozenset(self.link_ids))


def _resolve(index: GraphIndex, refs: tuple[str, ...]) -> set[str]:
    """Map framing refs -- ids, names, or aliases -- onto thing ids.

    The prompt says "IL-6", the graph says `t7`. Accepting both is the whole
    reason a clinician can steer this without reading the JSON.
    """
    if not refs:
        return set()
    wanted = {r.strip().lower() for r in refs}
    out: set[str] = set()
    for thing in index.things.values():
        names = {thing.id.lower(), thing.name.lower(), *(a.lower() for a in thing.aliases)}
        if names & wanted:
            out.add(thing.id)
    return out


def _excluded(index: GraphIndex, params: Params) -> set[str]:
    return _resolve(index, params.framing.exclude)


def _seed_ok(index: GraphIndex, thing_id: str, params: Params) -> bool:
    t = params.traversal
    thing = index.things.get(thing_id)
    if thing is None or thing.mentions < t.min_mentions:
        return False
    if thing_id in _excluded(index, params):
        return False
    anchors = _resolve(index, params.framing.anchors)
    if anchors and thing_id not in anchors:
        return False
    if t.metapaths:
        # A metapath fixes the seed kind; the loose filter would only argue
        # with it.
        return any(mp and thing.kind == mp[0] for mp in t.metapaths)
    return not t.seed_kinds or thing.kind in t.seed_kinds


def _target_ok(index: GraphIndex, thing_id: str, params: Params) -> bool:
    t = params.traversal
    thing = index.things.get(thing_id)
    if thing is None or thing.mentions < t.min_mentions:
        return False
    if thing_id in _excluded(index, params):
        return False
    targets = _resolve(index, params.framing.targets)
    if params.framing.mode == "closed" and targets and thing_id not in targets:
        return False
    if t.metapaths:
        return any(mp and thing.kind == mp[-1] for mp in t.metapaths)
    return not t.target_kinds or thing.kind in t.target_kinds


def _path_ok(index: GraphIndex, path: tuple[Edge, ...] | list[Edge], params: Params) -> bool:
    """Interior-node gates the per-edge check cannot see.

    ``walk`` judges edges; exclusions and the mentions floor are properties of
    the *node* the path passes through. A chain routed through the one thing a
    clinician said to route around is not improved by every hop being confident.
    """
    excluded = _excluded(index, params)
    floor = params.traversal.min_mentions
    for edge in path:
        thing = index.things.get(edge.dst)
        if edge.dst in excluded or edge.src in excluded:
            return False
        if thing is None or thing.mentions < floor:
            return False
    return True


def _shortest_path(
    index: GraphIndex, a: str, b: str, params: Params
) -> tuple[tuple[Edge, ...], float]:
    """Shortest confident path from a to b, with its weight, or empty.

    Used to give gap candidates the same evidence spine a chain candidate has:
    a gap with no path between its endpoints is a much weaker proposal than one
    the graph already almost connects.
    """
    for path, weight in index.walk(a, params.traversal):
        if path[-1].dst == b:
            return tuple(path), weight
    return (), 0.0


def _gap_closures(index: GraphIndex, params: Params) -> list[Candidate]:
    out: list[Candidate] = []
    for gap in index.graph.gaps:
        if len(gap.missing) != 2:
            continue
        a, b = gap.missing
        if a not in index.things or b not in index.things:
            continue
        if params.motifs.require_unstated and index.links_between(a, b):
            # A gap whose pair now has a link is stale -- either a later round
            # promoted it (which is `test_gap` working as designed) or the graph builder
            # is contradicting itself. Either way the "missing" relation is
            # stated, so proposing it is a restatement, and letting it through
            # would cost a real candidate its slot in selection.
            continue
        if not (_seed_ok(index, a, params) and _target_ok(index, b, params)):
            if not (_seed_ok(index, b, params) and _target_ok(index, a, params)):
                continue
        path, weight = _shortest_path(index, a, b, params)
        if path and not _path_ok(index, path, params):
            path, weight = (), 0.0
        out.append(
            Candidate(
                id=f"H-{gap.id}",
                motif="gap_closure",
                subject=a,
                object=b,
                path=path,
                gap_id=gap.id,
                tags=("searched" if gap.searched_in_round else "unsearched",),
                note=gap.note,
                weight=weight,
            )
        )
    return out


def _transitive_chains(index: GraphIndex, params: Params) -> list[Candidate]:
    t = params.traversal
    out: list[Candidate] = []
    for seed in index.things:
        if not _seed_ok(index, seed, params):
            continue
        for path, weight in index.walk(seed, t):
            if len(path) < 2:
                continue  # a single hop is a stated fact, not a hypothesis
            end = path[-1].dst
            if not _target_ok(index, end, params):
                continue
            if not _path_ok(index, path, params):
                continue
            if params.motifs.require_unstated and index.links_between(seed, end):
                continue  # somebody already stated it; nothing to propose
            gap = index.gap_between(seed, end)
            tags: list[str] = []
            if index.kind(seed) in INTERVENTION_KINDS and index.kind(end) == "disease":
                tags.append("repurposing")
            if any(not e.forward for e in path):
                tags.append("reversed_edge")
            out.append(
                Candidate(
                    id=f"H-chain-{seed}-{end}-{len(path)}",
                    motif="transitive_chain",
                    subject=seed,
                    object=end,
                    path=tuple(path),
                    gap_id=gap.id if gap else None,
                    tags=tuple(tags),
                    weight=weight,
                )
            )
            if len(out) >= t.max_candidates:
                return out
    return out


def _analogical_transfers(index: GraphIndex, params: Params) -> list[Candidate]:
    """X and Y look alike; X connects to Z; propose Y connects to Z too."""
    m = params.motifs
    out: list[Candidate] = []
    ids = [i for i in index.things if _seed_ok(index, i, params)]
    excluded = _excluded(index, params)
    for x, y in combinations(sorted(ids), 2):
        if m.analogy_same_kind_only and index.kind(x) != index.kind(y):
            continue  # analogy across kinds is usually a category error
        # Excluded things are dropped from the similarity basis, not just from
        # the endpoints. "Route around this node" has to mean it plays no part
        # in the argument either -- an analogy justified by the one hub the
        # clinician told us to ignore is exactly what they asked not to see.
        nx = index.neighbor_ids(x) - excluded
        ny = index.neighbor_ids(y) - excluded
        shared = nx & ny
        if len(shared) < m.analogy_min_shared:
            continue
        union = nx | ny
        # Raw overlap rewards hubs -- two promiscuous nodes share many
        # neighbours while being nothing alike. Jaccard normalises for that.
        jaccard = len(shared) / len(union) if union else 0.0
        if jaccard < m.analogy_min_jaccard:
            continue
        # Both directions. `combinations` yields each pair once, but the
        # analogy is not symmetric in what it *proposes*: the interesting
        # transfer is whichever thing has the edge the other lacks, and which
        # of the two that is has nothing to do with id order. Testing one
        # direction silently drops every hypothesis where the
        # lexicographically-later thing is the better-studied one.
        for donor, receiver in ((x, y), (y, x)):
            n_donor = nx if donor == x else ny
            n_receiver = ny if donor == x else nx
            for z in sorted(n_donor - n_receiver - {receiver}):
                if not _target_ok(index, z, params):
                    continue
                if m.require_unstated and index.links_between(receiver, z):
                    continue
                bridge = next(
                    (e for e in index.neighbors(donor) if e.dst == z), None
                )
                if bridge is None or not index.edge_ok(bridge, params.traversal):
                    continue
                out.append(
                    Candidate(
                        id=f"H-analog-{receiver}-{z}-via-{donor}",
                        motif="analogical_transfer",
                        subject=receiver,
                        object=z,
                        path=(bridge,),
                        analogues=(donor, *sorted(shared)),
                        gap_id=(g.id if (g := index.gap_between(receiver, z)) else None),
                        note=(
                            f"{index.name(receiver)} shares {len(shared)} neighbours "
                            f"with {index.name(donor)} (Jaccard {jaccard:.2f}), which "
                            f"is linked to {index.name(z)}"
                        ),
                        # The bridge is the donor's edge, not the receiver's: it
                        # makes the analogy concrete but is evidence about the
                        # donor. Discount it by how alike the pair actually is.
                        weight=index.path_weight((bridge,), params.traversal) * jaccard,
                    )
                )
                if len(out) >= params.traversal.max_candidates:
                    return out
    return out


def _reachable_from_anchors(index: GraphIndex, params: Params) -> set[str] | None:
    """Nodes the framing actually put in scope, or None when framing is open.

    A condition split is about a *link*, not a path, so the seed/target gates
    that shape the other motifs do not apply to it. Without this, asking a
    focused question still returns every disagreeing link in the graph -- true,
    but not an answer to what was asked.
    """
    anchors = _resolve(index, params.framing.anchors)
    if not anchors:
        return None
    reachable = set(anchors)
    for anchor in anchors:
        for path, _ in index.walk(anchor, params.traversal):
            reachable.update(e.dst for e in path)
    return reachable


def _condition_splits(index: GraphIndex, params: Params) -> list[Candidate]:
    out: list[Candidate] = []
    excluded = _excluded(index, params)
    in_scope = _reachable_from_anchors(index, params)
    for link in index.graph.links:
        if link.state != "disagreed":
            continue
        if link.src not in index.things or link.dst not in index.things:
            continue
        if {link.src, link.dst} & excluded:
            continue
        if any(
            index.things[t].mentions < params.traversal.min_mentions
            for t in (link.src, link.dst)
        ):
            continue
        if in_scope is not None and not {link.src, link.dst} & in_scope:
            continue
        conditions = index.conditions_for(link)
        if params.motifs.condition_split_requires_where and len(conditions) < 2:
            continue  # "maybe it's conditions" with no condition in hand
        edge = Edge(
            link_id=link.id, src=link.src, dst=link.dst, how=link.how, forward=True
        )
        out.append(
            Candidate(
                id=f"H-cond-{link.id}",
                motif="condition_split",
                subject=link.src,
                object=link.dst,
                path=(edge,),
                focus_link_id=link.id,
                conditions=tuple(conditions),
                tags=("conditions_stated",) if len(conditions) > 1 else ("conditions_missing",),
                note=link.why or "",
                weight=index.path_weight((edge,), params.traversal),
            )
        )
    return out


_GENERATORS = {
    "gap_closure": _gap_closures,
    "transitive_chain": _transitive_chains,
    "analogical_transfer": _analogical_transfers,
    "condition_split": _condition_splits,
}


def enumerate_candidates(
    index: GraphIndex,
    params: Params,
    focus_thing_id: str | None = None,
) -> list[Candidate]:
    """Every allowed structural hypothesis, optionally bound to one graph node.

    The focus filter is applied before the global candidate cap. Filtering a
    completed slate would let an unrelated dense neighbourhood consume the cap
    and incorrectly report that the requested focus had no candidates.
    """
    seen: set[tuple] = set()
    out: list[Candidate] = []
    for motif in params.motifs.enabled:
        generator = _GENERATORS.get(motif)
        if generator is None:
            continue
        for candidate in generator(index, params):
            if focus_thing_id and focus_thing_id not in candidate.node_ids():
                continue
            key = candidate.key()
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
            if len(out) >= params.traversal.max_candidates:
                return out
    return out
