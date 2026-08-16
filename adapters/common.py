"""What every adapter shares: how to read the core's output.

An adapter consumes ``hypothesis.json`` documents and produces something the
core deliberately does not — a markdown report, a UI payload, a valuation
brief. The rules are the same for all of them:

1. **Read documents, never the graph.** An adapter that opens the knowledge
   graph can state something the hypothesis was never checked against, and the
   audit trail silently stops being one.
2. **Never call a model.** Adapters are pure functions of their input. That is
   what makes them re-runnable over a saved document at no cost.
3. **Add no claim.** An adapter may drop detail, reorder it, or render it
   differently. It may not assert anything the document does not carry.
4. **Preserve the warnings.** Failure badges, halted verifications,
   error-level validation issues and the absence-of-evidence notice must
   survive every rendering. A view changes the form, never the safety.

The core emits exactly one hypothesis per run. Adapters accept **one or more**
documents, because some of their work is comparative by nature: a table ranks,
an SVG shows two hypotheses converging on a shared node, a valuation groups two
labels onto one molecule. Feeding several means running the core several times
and collecting the results -- which is the honest shape, since each document
then carries the provenance of the run that produced it.

``Bundle`` is that collection. It is a *local* convenience, not a contract: the
core has no such type, and nothing an adapter builds here may travel back.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hyp_gen.hypothesis import Ask, Hypothesis, HypothesisDocument, Provenance


class Bundle(BaseModel):
    """One or more hypothesis documents, viewed together.

    Provenance is taken from the first document and the rest are checked
    against it: rendering hypotheses from different graphs, or from runs at
    different craziness settings, into one table would put numbers side by side
    that do not mean the same thing.
    """

    model_config = ConfigDict(extra="allow")

    provenance: Provenance
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    asks: list[Ask] = Field(default_factory=list)

    # The renderers were written against a run-shaped object and read these
    # directly; keeping the names means an adapter's internals do not have to
    # know a document from a run.
    @property
    def graph_id(self) -> str:
        return self.provenance.graph_id

    @property
    def round(self) -> int:
        return self.provenance.round

    @property
    def question(self) -> str:
        return self.provenance.question

    @property
    def generated_at(self) -> str | None:
        return self.provenance.generated_at

    @property
    def params(self) -> dict:
        return self.provenance.params

    @property
    def coverage(self) -> dict:
        return self.provenance.coverage

    @property
    def counts(self) -> dict[str, int]:
        return self.provenance.counts

    @classmethod
    def of(cls, documents: list[HypothesisDocument]) -> Bundle:
        if not documents:
            raise ValueError("an adapter needs at least one hypothesis document")

        first = documents[0].provenance
        for doc in documents[1:]:
            if doc.provenance.graph_id != first.graph_id:
                raise ValueError(
                    "refusing to bundle hypotheses from different graphs: "
                    f"{first.graph_id} and {doc.provenance.graph_id}. Their scores are "
                    "not comparable and a shared header would claim they are."
                )

        seen: set[str] = set()
        asks: list[Ask] = []
        for doc in documents:
            for ask in doc.asks:
                key = f"{ask.ask}:{ask.target}"
                if key not in seen:
                    seen.add(key)
                    asks.append(ask)

        return cls(
            provenance=first,
            hypotheses=[doc.hypothesis for doc in documents],
            asks=asks,
        )


def load(*paths: str | Path) -> Bundle:
    """Read one or more ``hypothesis.json`` files into a bundle.

    A path may be a directory, in which case every ``*.json`` in it that parses
    as a hypothesis document is taken and the rest ignored -- a run directory
    usually holds other things. ``-`` reads one document from stdin.
    """
    documents: list[HypothesisDocument] = []
    for path in paths:
        if str(path) == "-":
            # `hypgen --graph g.json | hypreport -` with no file in between.
            documents.append(HypothesisDocument.model_validate(json.loads(sys.stdin.read())))
            continue
        path = Path(path)
        candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            try:
                documents.append(
                    HypothesisDocument.model_validate(json.loads(candidate.read_text()))
                )
            except Exception:
                if not path.is_dir():
                    raise
    if not documents:
        raise ValueError(f"no hypothesis documents found in {[str(p) for p in paths]}")
    return Bundle.of(documents)


# -- the safety helpers every adapter must use -----------------------------
#
# These four decide how a failure looks. They live here, not in one adapter,
# because rule 4 above is only real if every adapter renders a failure the same
# way -- an adapter that formats its own halt note is one refactor away from
# quietly dropping it. ``tests/test_adapters.py`` checks that each adapter's
# output still carries these signals.

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")

_STRUCTURAL_CODES = {"unknown_thing", "unknown_link", "broken_path", "already_stated"}


def _clip(text: str, limit: int) -> str:
    """Whole sentences up to ``limit``, marked with … if anything was dropped."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    kept = ""
    for sentence in _SENTENCE_END.split(text):
        if kept and len(kept) + len(sentence) + 1 > limit:
            break
        if not kept:
            kept = sentence
            if len(kept) > limit:
                # One sentence longer than the budget: cut on a word so the
                # clip mark lands somewhere a reader can see it.
                kept = kept[:limit].rsplit(" ", 1)[0]
                break
        else:
            kept = f"{kept} {sentence}"
    return f"{kept.rstrip()} …" if kept != text else text


def _failure_badges(hypothesis: Hypothesis) -> list[str]:
    errors = [i for i in hypothesis.issues if i.severity == "error"]
    badges: list[str] = []
    if any(i.code in _STRUCTURAL_CODES for i in errors):
        badges.append("**BLOCKED — not articulated**")
    if any(i.code == "illegal_citation" for i in errors):
        badges.append("**CITATION REJECTED — cites evidence it was not shown**")
    if errors and not badges:
        badges.append("**INVALID — see validation**")
    return badges


def _halt_note(hypothesis: Hypothesis) -> str:
    """The one sentence a halted verification must never be read without.

    Gates below a halt did not run. A view that shows the passes and omits the
    halt turns a partial verification into a clean-looking one, so every mode
    calls this rather than formatting its own.
    """
    verification = hypothesis.verification
    if not (verification and verification.halted_at):
        return ""
    halted = verification.halted_at
    gate = verification.gate(halted)
    return (
        f"Verification stopped at **{halted}**: {gate.summary if gate else ''} "
        "Every gate below it was not run, and none of them should be read as passed."
    )


def _flags(hypothesis: Hypothesis) -> list[str]:
    """Compact flags for views with no room for a badge line, e.g. a table row."""
    flags = [b.replace("*", "").split(" — ")[0] for b in _failure_badges(hypothesis)]
    verification = hypothesis.verification
    if verification and verification.halted_at:
        flags.append(f"HALTED at {verification.halted_at}")
    return flags

