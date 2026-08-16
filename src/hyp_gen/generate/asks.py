"""Turn each hypothesis's weakest point into a request back to the graph builder.

This is the closed loop. A hypothesis that cannot say what evidence would move
it is a guess; one that names the exact link to resolve or gap to test is a
plan. The asks are emitted in the request shape from the schema contract, keyed
by id rather than prose, so the graph builder can act on them without interpretation.

Priority is deliberately *not* "resolve the weakest link". It is "resolve the
link whose resolution would change the most", which is the weakest link that
the hypothesis actually depends on. For a chain aggregated by weakest-link that
is the same thing; for a gap it is the gap itself.
"""

from __future__ import annotations

from hyp_gen.generate.candidates import Candidate
from hyp_gen.graph import GraphIndex
from hyp_gen.params import Params
from hyp_gen.hypothesis import Ask
from hyp_gen.generate.scoring import Scores


def asks_for(
    index: GraphIndex,
    candidate: Candidate,
    scores: Scores,
    params: Params,
    hypothesis_id: str,
    rank_position: int = 1,
) -> list[Ask]:
    loop = params.loop
    graph_id = index.graph.graph_id
    out: list[Ask] = []
    if not loop.enabled:
        return out

    # The load-bearing gap. Testing it either promotes it to a real link (the
    # hypothesis becomes a known fact and is retired) or hardens it (searched
    # and not found is a far stronger novelty claim than never searched).
    # Only worth a round trip for hypotheses near the top of the record: a gap
    # under hypothesis 11 is not what the next search should spend itself on.
    if candidate.gap_id and candidate.gap_id in index.gaps:
        gap = index.gaps[candidate.gap_id]
        if gap.searched_in_round is None and rank_position <= loop.test_gap_when_ranked_above:
            out.append(
                Ask(
                    graph_id=graph_id,
                    ask="test_gap",
                    target=gap.id,
                    depth=loop.depth,
                    reason=(
                        "novelty of this hypothesis rests on nobody having stated "
                        "this pair, and nobody has looked yet"
                    ),
                    for_hypothesis=hypothesis_id,
                )
            )

    # The weakest link the hypothesis leans on. Under weakest-link aggregation
    # this is literally the number that would move.
    if scores.per_link:
        weakest = min(scores.per_link, key=lambda l: l.support)
        link = index.links.get(weakest.link_id)
        if link is not None and weakest.support < loop.resolve_link_below_confidence:
            if link.state == "disagreed":
                reason = (
                    f"this step disagrees across studies ({weakest.yes_count} yes / "
                    f"{weakest.no_count} no) and the hypothesis cannot be graded "
                    "until the conditions are separated"
                )
            elif link.state == "single_source":
                reason = (
                    "this step rests on a single source, so the chain's "
                    "independence is unestablished"
                )
            else:
                reason = (
                    f"weakest step in the chain (recomputed support "
                    f"{weakest.support:.2f}) and the hypothesis is only as strong "
                    "as this link"
                )
            out.append(
                Ask(
                    graph_id=graph_id,
                    ask="resolve_link",
                    target=link.id,
                    depth=loop.depth,
                    reason=reason,
                    for_hypothesis=hypothesis_id,
                )
            )

    # A central but under-connected endpoint: heavily mentioned, barely linked.
    # That shape means the search read about it without extracting its
    # relationships, which is a retrieval hole rather than a knowledge hole.
    for tid in (candidate.subject, candidate.object):
        thing = index.things.get(tid)
        if thing is None:
            continue
        if (
            thing.mentions >= loop.expand_node_min_mentions
            and index.degree(tid) <= loop.expand_node_max_degree
        ):
            out.append(
                Ask(
                    graph_id=graph_id,
                    ask="expand_node",
                    target=tid,
                    depth=loop.depth,
                    reason=(
                        f"{thing.name} appears in {thing.mentions} papers but has "
                        f"{index.degree(tid)} links — likely under-extracted"
                    ),
                    for_hypothesis=hypothesis_id,
                )
            )
    return out


def dedupe(asks: list[Ask]) -> list[Ask]:
    """One ask per target. The graph builder takes one ask per request anyway, and the
    same link being weak for three hypotheses is one piece of work, not three.
    """
    best: dict[tuple[str, str], Ask] = {}
    order = {"exhaustive": 3, "deep": 2, "standard": 1, "quick": 0}
    for ask in asks:
        key = (ask.ask, ask.target)
        current = best.get(key)
        if current is None or order[ask.depth] > order[current.depth]:
            best[key] = ask
    return list(best.values())
