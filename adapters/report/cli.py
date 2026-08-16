"""    hypreport runs/first/hypothesis.json --mode prose --out runs/first"""

from __future__ import annotations

import argparse
from pathlib import Path

from adapters.common import load
from adapters.report.render import FILENAMES, MODE_NAMES, to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hypreport",
        description="Render hypothesis.json as markdown. Reads the document only.",
    )
    parser.add_argument(
        "documents",
        nargs="+",
        type=Path,
        metavar="HYPOTHESIS.JSON",
        help="one or more hypothesis.json files, or a directory holding them",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=MODE_NAMES,
        metavar="MODE",
        help=(
            "repeatable: prose (report.md, the default) | table "
            "(report-table.md, one row each) | trace (report-trace.md, the "
            "graph walk with each edge's evidence) | full (report-full.md, "
            "claims, gate tables and verbatim sources)"
        ),
    )
    parser.add_argument("--out", type=Path, metavar="DIR", help="write files here instead of stdout")
    args = parser.parse_args(argv)

    bundle = load(*args.documents)
    modes = args.mode or ["prose"]

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for mode in modes:
            path = args.out / FILENAMES[mode]
            path.write_text(to_markdown(bundle, mode=mode))
            print(f"wrote {path} ({mode})")
        return 0

    print("\n\n".join(to_markdown(bundle, mode=m) for m in modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
