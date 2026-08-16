"""The evidence pack: the only thing the model is allowed to see per candidate.

Two jobs. First, it bounds the model's world to the subgraph the enumerator
selected, so an articulated hypothesis cannot quietly import background
knowledge and present it as a graph finding. Second, it fixes the set of legal
citation ids -- anything the model cites that is not in the pack is rejected by
``validate.py`` rather than believed.

Verbatim quotes are included because they are the bottom of the audit trail. A
reader who distrusts a claim should be able to land on the exact sentence a
human wrote, without leaving the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hyp_gen.generate.candidates import Candidate
from hyp_gen.graph import GraphIndex
from hyp_gen.generate.scoring import Scores


@dataclass
class EvidencePack:
    candidate: Candidate
    things: dict[str, dict] = field(default_factory=dict)
    links: dict[str, dict] = field(default_factory=dict)
    findings: dict[str, dict] = field(default_factory=dict)
    papers: dict[str, dict] = field(default_factory=dict)
    gap: dict | None = None
    caveats: list[str] = field(default_factory=list)

    def legal_ids(self) -> set[str]:
        return {*self.things, *self.links, *self.findings, *self.papers} | (
            {self.gap["id"]} if self.gap else set()
        )

    def to_prompt(self) -> str:
        lines: list[str] = []
        lines.append("THINGS")
        for tid, t in self.things.items():
            alias = f" (aka {', '.join(t['aliases'])})" if t["aliases"] else ""
            lines.append(f"  {tid} [{t['kind']}] {t['name']}{alias} — {t['mentions']} papers")

        lines.append("\nLINKS (the graph's summary of its own findings)")
        for lid, l in self.links.items():
            lines.append(
                f"  {lid}: {l['from_name']} --{l['how']}--> {l['to_name']}"
                f" | state={l['state']} basis={l['basis']}"
                f" | yes={l['yes']} no={l['no']} no_effect={l['no_effect']}"
                f" | recomputed_support={l['recomputed_support']}"
            )
            if l["why"]:
                lines.append(f"      why disagreed: {l['why']}")
            if l["conditions"]:
                lines.append(f"      conditions seen: {'; '.join(l['conditions'])}")

        lines.append("\nFINDINGS (raw claims, with the exact source sentence)")
        for fid, f in self.findings.items():
            marks = []
            if f["hedged"]:
                marks.append("hedged")
            if not f["is_own_result"]:
                marks.append("citing others")
            mark = f" [{', '.join(marks)}]" if marks else ""
            lines.append(
                f"  {fid} ({f['paper']}, says={f['says']}, where={f['where'] or 'unstated'})"
                f"{mark}: \"{f['quote']}\""
            )

        lines.append("\nPAPERS")
        for pid, p in self.papers.items():
            pre = " [preprint]" if p["is_preprint"] else ""
            lines.append(
                f"  {pid}: {p['first_author'] or '?'} {p['year'] or '?'},"
                f" {p['study_type']}{pre} — {p['title']}"
            )

        if self.gap:
            lines.append("\nGAP FLAGGED BY THE GRAPH")
            searched = (
                f"searched in round {self.gap['searched_in_round']}"
                if self.gap["searched_in_round"]
                else "never searched for"
            )
            lines.append(f"  {self.gap['id']}: {self.gap['note']} ({searched})")

        if self.caveats:
            lines.append("\nCAVEATS ON THIS GRAPH")
            for caveat in self.caveats:
                lines.append(f"  - {caveat}")
        return "\n".join(lines)


def _caveats(index: GraphIndex, candidate: Candidate, scores: Scores) -> list[str]:
    out: list[str] = []
    cov = index.graph.coverage
    if cov.truncated:
        out.append(
            f"The search is a sample, not the literature: {cov.read} of "
            f"{cov.found} results were read"
            + (f" (hit {cov.limits.hit_limit})" if cov.limits.hit_limit else "")
            + ". Absence of a link is weak evidence of absence."
        )
    if cov.depth == "quick":
        out.append(
            "Depth is `quick`, which reads only the first page of results. At "
            "this tier absence means unknown, never 'no evidence'."
        )
    if cov.no_quote_discarded:
        out.append(
            f"{cov.no_quote_discarded} extracted claims were dropped for having "
            "no verbatim sentence, so the graph under-reports slightly."
        )
    if any(not e.forward for e in candidate.path):
        out.append(
            "This chain traverses at least one link against its stated "
            "direction. Do not assert the reverse relation as established."
        )
    if scores.absence_reliability < 0.5 and candidate.gap_id:
        out.append(
            "Novelty here rests on a gap, but this graph is not entitled to "
            "strong absence claims. Treat 'nobody has shown X' as 'this search "
            "did not surface X'."
        )
    return out


def build_pack(index: GraphIndex, candidate: Candidate, scores: Scores) -> EvidencePack:
    support_by_link = {l.link_id: l for l in scores.per_link}
    pack = EvidencePack(candidate=candidate)

    for tid in candidate.node_ids():
        thing = index.things.get(tid)
        if thing is None:
            continue
        pack.things[tid] = {
            "name": thing.name,
            "kind": thing.kind,
            "aliases": thing.aliases,
            "mentions": thing.mentions,
        }

    for lid in candidate.link_ids:
        link = index.links.get(lid)
        if link is None:
            continue
        recomputed = support_by_link.get(lid)
        pack.links[lid] = {
            "from": link.src,
            "to": link.dst,
            "from_name": index.name(link.src),
            "to_name": index.name(link.dst),
            "how": link.how,
            "state": link.state,
            "basis": link.basis,
            "why": link.why,
            "yes": link.yes,
            "no": link.no,
            "no_effect": link.no_effect,
            "stated_confidence": link.confidence.overall,
            "recomputed_support": recomputed.support if recomputed else None,
            "conditions": index.conditions_for(link),
        }
        for finding in index.findings_for(link):
            pack.findings[finding.id] = {
                "from": finding.src,
                "to": finding.dst,
                "how": finding.how,
                "says": finding.says,
                "quote": finding.quote,
                "paper": finding.paper,
                "where": finding.where,
                "is_own_result": finding.is_own_result,
                "hedged": finding.hedged,
                "confidence": finding.confidence,
            }
            paper = index.paper_for(finding)
            if paper is not None:
                pack.papers[paper.id] = {
                    "title": paper.title,
                    "year": paper.year,
                    "journal": paper.journal,
                    "doi": paper.doi,
                    "first_author": paper.first_author,
                    "study_type": paper.study_type,
                    "is_preprint": paper.is_preprint,
                }

    if candidate.gap_id and candidate.gap_id in index.gaps:
        gap = index.gaps[candidate.gap_id]
        pack.gap = {
            "id": gap.id,
            "missing": gap.missing,
            "implied_by": gap.implied_by,
            "note": gap.note,
            "confidence": gap.confidence,
            "searched_in_round": gap.searched_in_round,
        }

    pack.caveats = _caveats(index, candidate, scores)
    return pack
