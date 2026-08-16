"""The model stages: articulate, attack, compare, evolve.

This is the co-scientist generate/debate/evolve loop with one rule added:
nothing here may introduce a fact. Every stage sees only the evidence pack the
deterministic stages built, is told that ids outside it are illegal, and is
checked afterwards by ``validate.check_citations``. The instruction sets the
expectation; the check is what makes it true.

Articulation and critique are separate calls with separate system prompts on
purpose. One pass asked to both propose and criticise produces a hypothesis
pre-softened to survive its own review; a second pass told only to break things
finds more. Critics are further diversified by *lens* rather than by
resampling -- three identical refuters mostly agree with each other, while a
mechanism critic and an evidence critic fail on different things.
"""

from __future__ import annotations

from hyp_gen.generate.evidence import EvidencePack
from hyp_gen.reasoning.llm import Judge
from hyp_gen.params import Params
from hyp_gen.hypothesis import Articulation, Comparison, Critique, Verdict

_MOTIF_BRIEF = {
    "gap_closure": (
        "The graph's own analysis flagged this pair as implied by its links but "
        "never directly stated. Propose the missing relation."
    ),
    "transitive_chain": (
        "These things are connected only through intermediates. Propose the "
        "composed relation, and be explicit about which step is the weak one."
    ),
    "analogical_transfer": (
        "Two things share several neighbours; one has a relation the other "
        "lacks. Propose the transfer, and say what would make the analogy fail."
    ),
    "condition_split": (
        "This link disagrees with itself across studies. The usual cause is "
        "different experimental conditions, not conflict. Propose the condition "
        "that reconciles both results — a hypothesis about WHEN the relation "
        "holds, not whether."
    ),
}

LENS_BRIEF = {
    "mechanism": (
        "Attack the mechanism. Does the proposed causal story actually follow "
        "from the links, in the direction they are stated? Is a correlation "
        "being read as causation? Is a step missing that the hypothesis needs?"
    ),
    "evidence": (
        "Attack the evidence. For each claim, is the cited evidence the kind "
        "that could establish it? Watch for single-source links, one research "
        "group counted twice, hedged or secondhand findings carrying a "
        "confident conclusion, and test-tube results generalised to patients."
    ),
    "testability": (
        "Attack the testability. Is the statement specific enough to be wrong? "
        "Would the proposed falsifier actually falsify it, or is it compatible "
        "with the hypothesis being false? Is the decisive experiment decisive?"
    ),
    "novelty": (
        "Attack the novelty. Does the pack already state this outright? Is the "
        "claim of novelty resting on absence in a search that was truncated or "
        "shallow? Check the caveats before accepting 'nobody has shown this'."
    ),
    "conditions": (
        "Attack the conditions. Do the findings being combined come from the "
        "same system? Check `where` on every finding: two results that appear "
        "to agree in different systems may not agree at all."
    ),
}

ARTICULATE_SYSTEM = """\
You turn a structural pattern found in a knowledge graph into one precise, \
testable scientific hypothesis.

Hard rules:
- The evidence pack below is your entire world. Do not use outside knowledge \
about these entities, and do not assert anything the pack does not support.
- Cite by id. Every id you write must appear in the pack: link ids, finding \
ids, paper ids, thing ids, or the gap id. Inventing an id invalidates the run.
- A step of reasoning the graph does not state is legitimate — mark that claim \
`inferred: true` and say what it assumes. What is not legitimate is presenting \
it as something the graph states.
- The hypothesis must be one the graph does NOT already contain. If the pack \
turns out to state the relation outright, say so plainly in `novel_because` \
rather than dressing it up as new.
- Prefer the specific over the safe. "Compound X inhibits Y in fibrotic tissue \
but not in healthy tissue" is worth testing; "X may play a role in Y" is not.
- The falsifier must be a single observation that would kill the hypothesis, \
and the decisive experiment must be the cheapest thing that discriminates it \
from the obvious duller alternative.
"""

CRITIQUE_SYSTEM = """\
You are reviewing a hypothesis generated from a knowledge graph. Your job is to \
break it, not to improve it.

Work only from the evidence pack. For each claim, decide whether the pack \
actually supports it, and cite the ids that decide it.

Be concrete. "Weak evidence" is not an objection; "L4 rests on one test-tube \
result from the same group as L2, so the independence this hypothesis assumes \
is not there" is. If the evidence does support a claim, say so — a critic that \
objects to everything is as useless as one that objects to nothing.
"""

COMPARE_SYSTEM = """\
You are judging which of two hypotheses is more worth taking to the bench, \
given only the evidence packs shown.

Judge on: whether the evidence supports the claim, whether it says something \
the graph does not already say, and whether the proposed experiment would \
actually settle it. A safe restatement of known biology loses to a risky claim \
with a clean falsifier; a fluent story with no evidence loses to both.

Cite the ids that decided it. Ids outside the two packs are illegal.
"""

EVOLVE_SYSTEM = """\
You are revising a hypothesis in response to specific criticism.

The same rules apply as when it was written: the evidence pack is your entire \
world, every id must come from it, and inferred steps must be marked. You may \
narrow the claim, add a condition, or restate the mechanism — you may not \
rescue it by making it vaguer. If the criticism is fatal and no honest revision \
survives, say so in `novel_because` and leave the statement as the narrowest \
version the evidence supports.
"""

EVOLVE_OPERATORS = {
    "specialise": (
        "Narrow the hypothesis until the strongest objection no longer applies "
        "— usually by restricting it to the system, dose, or population the "
        "evidence actually covers."
    ),
    "combine": (
        "Fold in the alternative explanation the critic raised, so the "
        "hypothesis discriminates between the two rather than ignoring one."
    ),
    "invert_condition": (
        "Restate the hypothesis around the condition under which it fails. A "
        "claim about when something does NOT happen is often the sharper test."
    ),
}


def _user_prompt(pack: EvidencePack, extra: str = "") -> str:
    c = pack.candidate
    brief = _MOTIF_BRIEF.get(c.motif, "")
    head = (
        f"PATTERN: {c.motif}\n"
        f"SUBJECT: {c.subject} ({pack.things.get(c.subject, {}).get('name', c.subject)})\n"
        f"OBJECT: {c.object} ({pack.things.get(c.object, {}).get('name', c.object)})\n"
        f"HOPS: {c.hops}\n"
    )
    if c.path:
        chain = " -> ".join(
            f"{pack.things.get(e.src, {}).get('name', e.src)}"
            f" --[{e.how}{'' if e.forward else ', REVERSED'}]-> "
            f"{pack.things.get(e.dst, {}).get('name', e.dst)}"
            for e in c.path
        )
        head += f"CHAIN: {chain}\n"
    if c.conditions:
        head += f"CONDITIONS OBSERVED: {'; '.join(c.conditions)}\n"
    if c.analogues:
        names = ", ".join(pack.things.get(a, {}).get("name", a) for a in c.analogues)
        head += f"ANALOGUES: {names}\n"
    if c.note:
        head += f"GRAPH NOTE: {c.note}\n"
    return f"{brief}\n\n{head}\n{pack.to_prompt()}\n{extra}".strip()


def _render(articulation: Articulation) -> str:
    claims = "\n".join(
        f"  [{i}] {c.text}  (cites: {', '.join(c.cites) or 'none'}"
        f"{', INFERRED' if c.inferred else ''})"
        for i, c in enumerate(articulation.claims)
    )
    return (
        f"  statement: {articulation.statement}\n"
        f"  mechanism: {articulation.mechanism}\n"
        f"  novel because: {articulation.novel_because}\n"
        f"  falsifier: {articulation.falsifier}\n"
        f"  decisive experiment: {articulation.decisive_experiment}\n"
        f"  claims:\n{claims}\n"
        f"  assumptions: {'; '.join(articulation.assumptions) or 'none stated'}"
    )


def articulate(judge: Judge, pack: EvidencePack, params: Params) -> Articulation:
    budget = (
        f"\n\nDecompose into at most {params.ranking.max_claims_per_hypothesis} "
        "claims. Each one must be separately checkable against the pack — if you "
        "need more than that, the hypothesis is doing too much and should be "
        "narrowed rather than split further."
    )
    return judge.parse(
        system=ARTICULATE_SYSTEM,
        prompt=_user_prompt(pack, budget),
        schema=Articulation,
        effort=params.ranking.effort_articulate,
    )


def critique(
    judge: Judge,
    pack: EvidencePack,
    articulation: Articulation,
    lens: str,
    params: Params,
) -> Critique:
    lens_brief = LENS_BRIEF.get(lens, "Attack the hypothesis on any grounds.")
    extra = (
        f"\n\nYOUR LENS: {lens}\n{lens_brief}\n\n"
        f"HYPOTHESIS UNDER REVIEW\n{_render(articulation)}\n"
    )
    result = judge.parse(
        system=CRITIQUE_SYSTEM,
        prompt=_user_prompt(pack, extra),
        schema=Critique,
        effort=params.ranking.effort_critique,
    )
    result.lens = lens  # set by the harness; the model does not get to pick
    return result


def consensus(critiques: list[Critique], params: Params) -> Verdict | None:
    """Fold several lenses into one verdict.

    A single lens calling a hypothesis unsupported is information, not a
    ruling: a testability critic can legitimately reject something a mechanism
    critic accepts. It takes ``refute_threshold`` of them to drop it.
    """
    if not critiques:
        return None
    fatal = sum(1 for c in critiques if c.verdict in ("unsupported", "contradicted"))
    if fatal / len(critiques) >= params.ranking.refute_threshold:
        return "contradicted" if any(
            c.verdict == "contradicted" for c in critiques
        ) else "unsupported"
    if all(c.verdict == "supported" for c in critiques):
        return "supported"
    return "partly_supported"


def compare(
    judge: Judge,
    pack_a: EvidencePack,
    art_a: Articulation,
    pack_b: EvidencePack,
    art_b: Articulation,
) -> Comparison:
    # Built as one f-string rather than adjacent literals: implicit
    # concatenation binds tighter than `*`, so a bare `"=" * 60` in the middle
    # of a literal run multiplies everything before it instead of the divider.
    divider = "=" * 60
    prompt = (
        f"HYPOTHESIS A\n{_render(art_a)}\n\n"
        f"A'S EVIDENCE\n{pack_a.to_prompt()}\n\n"
        f"{divider}\n\n"
        f"HYPOTHESIS B\n{_render(art_b)}\n\n"
        f"B'S EVIDENCE\n{pack_b.to_prompt()}\n"
    )
    return judge.parse(system=COMPARE_SYSTEM, prompt=prompt, schema=Comparison)


def evolve(
    judge: Judge,
    pack: EvidencePack,
    articulation: Articulation,
    critiques: list[Critique],
    operator: str,
) -> Articulation:
    objections = "\n".join(
        f"  [{c.lens or 'general'}] {c.verdict}: {c.strongest_objection}"
        for c in critiques
    )
    leaps = "\n".join(
        f"  - {leap}" for c in critiques for leap in c.unsupported_leaps
    )
    extra = (
        f"\n\nOPERATOR: {operator}\n{EVOLVE_OPERATORS.get(operator, '')}\n\n"
        f"CURRENT HYPOTHESIS\n{_render(articulation)}\n\n"
        f"CRITICISM TO ANSWER\n{objections}\n"
        + (f"\nUNSUPPORTED LEAPS FLAGGED\n{leaps}\n" if leaps else "")
    )
    return judge.parse(
        system=EVOLVE_SYSTEM,
        prompt=_user_prompt(pack, extra),
        schema=Articulation,
    )
