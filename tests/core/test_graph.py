"""The graph layer: parsing, indexing, and the traversal the motifs sit on."""

from __future__ import annotations

import pytest

from hyp_gen.graph import Coverage, GraphIndex, KnowledgeGraph, Limits
from hyp_gen.params import TraversalParams


def test_loads_the_contract_shape(graph: KnowledgeGraph) -> None:
    assert graph.graph_id == "g_demo1"
    assert graph.round == 2
    assert {t.id for t in graph.things} >= {"t1", "t5"}
    # `from` and `to` are Python keywords in the wire format; the aliases have
    # to survive parsing or every downstream traversal is silently empty.
    link = next(l for l in graph.links if l.id == "L1")
    assert (link.src, link.dst) == ("t1", "t3")
    finding = next(f for f in graph.findings if f.id == "f1")
    assert (finding.src, finding.dst) == ("t1", "t3")


def test_unknown_fields_do_not_crash() -> None:
    """The graph builder owns the schema and will grow it. A new key is not our problem."""
    graph = KnowledgeGraph.model_validate(
        {
            "graph_id": "g_x",
            "things": [{"id": "t1", "name": "x", "kind": "gene", "vibe": "new"}],
            "links": [],
            "unexpected_top_level": 42,
        }
    )
    assert graph.things[0].name == "x"


def test_index_builds_both_directions(index: GraphIndex) -> None:
    forward = [e for e in index.neighbors("t1") if e.dst == "t3"]
    backward = [e for e in index.neighbors("t3") if e.dst == "t1"]
    assert forward and forward[0].forward is True
    assert backward and backward[0].forward is False
    assert index.links_between("t1", "t3") == index.links_between("t3", "t1")


def test_dangling_endpoints_are_skipped() -> None:
    graph = KnowledgeGraph.model_validate(
        {
            "graph_id": "g_x",
            "things": [{"id": "t1", "name": "a"}],
            "links": [{"id": "L1", "from": "t1", "how": "binds", "to": "t_missing"}],
        }
    )
    index = GraphIndex(graph)
    assert index.neighbors("t1") == []


def test_conditions_come_from_findings(index: GraphIndex) -> None:
    link = index.links["L6"]
    conditions = index.conditions_for(link)
    # Schema note 5: a `disagreed` link is usually two conditions, not a fight.
    assert len(conditions) == 2
    assert any("aged" in c for c in conditions)


@pytest.mark.parametrize(
    "depth,truncated,hit_limit,expected_range",
    [
        ("quick", False, None, (0.0, 0.0)),      # page one lies; absence means unknown
        ("standard", False, None, (0.4, 0.5)),
        ("deep", False, None, (0.75, 0.85)),
        ("exhaustive", False, None, (0.95, 1.0)),
        ("exhaustive", True, "max_papers", (0.45, 0.55)),
    ],
)
def test_absence_reliability_tracks_coverage(
    graph: KnowledgeGraph, depth, truncated, hit_limit, expected_range
) -> None:
    clone = graph.model_copy(deep=True)
    clone.coverage = Coverage(
        depth=depth, truncated=truncated, limits=Limits(hit_limit=hit_limit)
    )
    value = GraphIndex(clone).absence_reliability()
    assert expected_range[0] <= value <= expected_range[1]


def test_walk_respects_max_hops(index: GraphIndex) -> None:
    two = list(index.walk("t1", TraversalParams(max_hops=2)))
    three = list(index.walk("t1", TraversalParams(max_hops=3)))
    assert all(len(path) <= 2 for path, _ in two)
    assert max(len(path) for path, _ in three) == 3


def test_walk_respects_confidence_floor(index: GraphIndex) -> None:
    strict = list(index.walk("t8", TraversalParams(min_link_confidence=0.9)))
    assert strict == []


def test_walk_can_refuse_to_reverse_edges(index: GraphIndex) -> None:
    forward_only = list(
        index.walk("t6", TraversalParams(allow_edge_reversal=False, max_hops=3))
    )
    # t6 (PDGFR) is only ever an object, so with reversal off it is a dead end.
    assert forward_only == []
    with_reversal = list(index.walk("t6", TraversalParams(max_hops=3)))
    assert with_reversal


def test_negative_links_are_not_chained_by_default(graph: KnowledgeGraph) -> None:
    """"A does not do B, B does C" composes to nothing."""
    clone = graph.model_copy(deep=True)
    link = next(l for l in clone.links if l.id == "L6")
    link.no, link.yes = ["f7", "f6"], []  # tip it majority-negative
    index = GraphIndex(clone)

    assert index.is_negative(link)
    edge = next(e for e in index.neighbors("t9") if e.link_id == "L6")
    assert not index.edge_ok(edge, TraversalParams())
    assert index.edge_ok(edge, TraversalParams(allow_negative_edges=True))


def test_predicate_denylist_blocks_composition(index: GraphIndex) -> None:
    params = TraversalParams(predicates_deny=("drives",))
    assert not any(
        e.how == "drives" for path, _ in index.walk("t3", params) for e in path
    )


def test_hub_damping_penalises_busy_nodes(index: GraphIndex) -> None:
    """A path through a promiscuous node must be worth less than one through a
    sparse node, which is the entire point of DWPC."""
    through_hub = [e for e in index.neighbors("t3")][:1]
    undamped = index.dwpc(through_hub, damping=0.0)
    damped = index.dwpc(through_hub, damping=0.4)
    assert undamped == 1.0
    assert damped < undamped


def test_reversal_penalty_applies_per_hop(index: GraphIndex) -> None:
    params = TraversalParams(reversal_penalty=0.5, hub_damping=0.0)
    reversed_edge = next(e for e in index.neighbors("t3") if e.link_id == "L1")
    assert reversed_edge.forward is False
    assert index.path_weight([reversed_edge], params) == pytest.approx(0.5)


def test_metapath_constrains_the_kind_sequence(index: GraphIndex) -> None:
    params = TraversalParams(
        max_hops=3,
        metapaths=(("small_molecule", "process", "disease"),),
    )
    paths = list(index.walk("t1", params))
    assert paths
    for path, _ in paths:
        kinds = [index.kind("t1"), *(index.kind(e.dst) for e in path)]
        assert kinds == ["small_molecule", "process", "disease"]
