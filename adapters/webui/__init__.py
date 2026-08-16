"""hypothesis.json -> what a web front end needs.

Two artifacts, one input:

    payload.emit(bundle)   one JSON card per hypothesis: the walk as a string,
                           the metrics that order it, and `highlights` --
                           one-liners saying how the graph supports,
                           contradicts or qualifies the claim, each carrying
                           the ids it was built from so the UI can link a line
                           to its evidence.
    diagram.to_svg(bundle) a static SVG of the walks: edges coloured by what
                           the evidence says, nodes deduplicated so two
                           hypotheses crossing one node visibly converge.

See SCHEMA.md in this directory.
"""

from adapters.webui.diagram import to_svg
from adapters.webui.payload import Card, Highlight, WebPayload, emit

__all__ = ["Card", "Highlight", "WebPayload", "emit", "to_svg"]
