from __future__ import annotations

from pathlib import Path

import pytest

from hyp_gen.graph import GraphIndex, KnowledgeGraph
from hyp_gen.params import Params

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "knowledge-graph.json"


@pytest.fixture
def graph() -> KnowledgeGraph:
    """Function-scoped on purpose.

    ``GraphIndex`` holds references to the graph's own model objects, so any
    test that pokes a link to set up a case would otherwise leak that mutation
    into every later test in the session. Reloading is microseconds.
    """
    return KnowledgeGraph.load(FIXTURE)


@pytest.fixture
def index(graph: KnowledgeGraph) -> GraphIndex:
    return GraphIndex(graph)


@pytest.fixture
def params() -> Params:
    return Params()
