"""Shared by the adapter suites.

An adapter's real input is a directory of ``hypothesis.json`` files. These
tests short-circuit the disk round trip where what is under test is the
rendering rather than the loading -- ``test_adapter_clis.py`` covers the real
path, document files and all.
"""

from __future__ import annotations

from adapters.common import Bundle
from hyp_gen.pipeline import RunResult


def bundle(result: RunResult) -> Bundle:
    """Everything a run produced, viewed the way an adapter sees it."""
    return Bundle(
        provenance=result.provenance,
        hypotheses=result.hypotheses,
        asks=result.asks,
    )
