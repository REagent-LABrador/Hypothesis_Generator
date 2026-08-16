"""hyp_gen — a knowledge graph in, one hypothesis out.

The core flow, and nothing else:

    read the graph  ->  traverse it  ->  reason over what the traversal found
                    ->  check that reasoning back against the graph
                    ->  write one hypothesis

Two contracts bound it, both documented in this directory:

    graph.py       INPUT  — the knowledge graph accepted.  INPUT_SCHEMA.md
    hypothesis.py  OUTPUT — one ``HypothesisDocument``.    OUTPUT_SCHEMA.md

A run produces exactly one hypothesis: the one that ranked first. Many are
enumerated, scored and ranked on the way there, because that is how the winner
is known, but only the winner crosses the boundary.

    params.py    the knobs: profiles, the craziness dial, thresholds
    pipeline.py  the run itself, and ``RunResult.top()`` — the boundary

The stages are grouped by what they need and what they trust:

    generate/   deterministic pattern finding over the graph. No API key, no
                model call; this is what ``--dry-run`` executes end to end.
    reasoning/  the only place a model is called, and only for candidates that
                survived selection.
    checks/     everything ``reasoning/`` produced, evaluated back against the
                graph and the evidence pack it was shown.

Anything that turns a hypothesis into something else — a report, a UI payload,
an SVG, a valuation brief — is an **adapter** and lives outside this package,
in ``adapters/``. The core neither imports them nor knows they exist. That is
the point: if a hypothesis cannot be understood from ``hypothesis.json`` alone,
no adapter can rescue it.

Names re-exported below are the supported API; everything else is an
implementation detail and may move.
"""

from hyp_gen.checks.verify import GateContext
from hyp_gen.generate.candidates import Candidate, enumerate_candidates
from hyp_gen.generate.evidence import EvidencePack, build_pack
from hyp_gen.generate.scoring import (
    LinkSupport,
    Scores,
    score_all,
    score_candidate,
    score_link,
)
from hyp_gen.generate.select import pareto_front, select
from hyp_gen.graph import GraphIndex, KnowledgeGraph
from hyp_gen.hypothesis import (
    SCHEMA_VERSION,
    Articulation,
    Ask,
    Claim,
    Comparison,
    Critique,
    GateResult,
    Hypothesis,
    HypothesisDocument,
    Provenance,
    ValidationIssue,
    Verification,
)
from hyp_gen.params import PROFILES, Params
from hyp_gen.pipeline import Generator, RunResult
from hyp_gen.reasoning.llm import BudgetExceeded, Judge, RefusalError

__version__ = "0.1.0"

# The `verify` *function* is deliberately not re-exported here: callers should
# say which they mean -- `from hyp_gen.checks import verify` for the module,
# `from hyp_gen.checks.verify import verify` for the function.

__all__ = [
    "PROFILES",
    "SCHEMA_VERSION",
    "Articulation",
    "Ask",
    "BudgetExceeded",
    "Candidate",
    "Claim",
    "Comparison",
    "Critique",
    "EvidencePack",
    "GateContext",
    "GateResult",
    "Generator",
    "GraphIndex",
    "Hypothesis",
    "HypothesisDocument",
    "Judge",
    "KnowledgeGraph",
    "LinkSupport",
    "Params",
    "Provenance",
    "RefusalError",
    "RunResult",
    "Scores",
    "ValidationIssue",
    "Verification",
    "build_pack",
    "enumerate_candidates",
    "pareto_front",
    "score_all",
    "score_candidate",
    "score_link",
    "select",
]
