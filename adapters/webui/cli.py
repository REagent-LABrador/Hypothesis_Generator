"""    hypwebui runs/first/hypothesis.json --cards cards.json --svg traces.svg"""

from __future__ import annotations

import argparse
from pathlib import Path

from adapters.common import load
from adapters.webui.diagram import to_svg
from adapters.webui.payload import emit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hypwebui",
        description="Build the web UI payload and SVG trace from hypothesis.json.",
    )
    parser.add_argument(
        "documents",
        nargs="+",
        type=Path,
        metavar="HYPOTHESIS.JSON",
        help="one or more hypothesis.json files, or a directory holding them",
    )
    parser.add_argument("--cards", type=Path, metavar="FILE", help="write the card payload JSON here")
    parser.add_argument("--svg", type=Path, metavar="FILE", help="write the trace SVG here")
    args = parser.parse_args(argv)

    if not (args.cards or args.svg):
        parser.error("nothing to write: pass --cards, --svg, or both")

    bundle = load(*args.documents)

    if args.cards:
        args.cards.parent.mkdir(parents=True, exist_ok=True)
        args.cards.write_text(emit(bundle).model_dump_json(indent=2))
        print(f"wrote {args.cards}")

    if args.svg:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(to_svg(bundle))
        print(f"wrote {args.svg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
