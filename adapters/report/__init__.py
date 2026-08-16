"""hypothesis.json -> a report someone reads.

Four modes over the same input, because four different questions get asked of
one hypothesis:

    prose (default)  what is this idea, is it any good, what would kill it
    table            which of these should I look at first
    trace            where did this come from
    full             is the work correct

``to_markdown(bundle, mode=...)`` is the whole surface. See INPUT_SCHEMA.md and
OUTPUT_SCHEMA.md in this directory.
"""

from adapters.report.render import FILENAMES, MODE_NAMES, to_markdown

__all__ = ["FILENAMES", "MODE_NAMES", "to_markdown"]
