"""    hypvaluation runs/first/hypothesis.json --frame frame.json --out programs/

Start with ``--emit-frame-template``. The frame is the analyst's half of this
input and neither the graph nor this adapter can supply it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from adapters.common import load
from adapters.valuation.program import ProgramFrame, emit


def _frame(path: Path) -> ProgramFrame:
    raw = json.loads(path.read_text())
    # The template ships a `_README` explaining the four year fields; a caller
    # who filled the frame in and left the note there meant no harm by it.
    return ProgramFrame.model_validate({k: v for k, v in raw.items() if k != "_README"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hypvaluation",
        description="Turn hypothesis.json into program briefs for the ROI model.",
    )
    parser.add_argument(
        "documents",
        nargs="*",
        type=Path,
        metavar="HYPOTHESIS.JSON",
        help="one or more hypothesis.json files, or a directory holding them",
    )
    parser.add_argument("--frame", type=Path, metavar="FILE", help="analyst frame JSON")
    parser.add_argument("--out", type=Path, metavar="DIR", help="directory for *.program.json")
    parser.add_argument(
        "--emit-frame-template",
        type=Path,
        metavar="FILE",
        help="write a starter frame and exit; fill in the four null year fields yourself",
    )
    args = parser.parse_args(argv)

    if args.emit_frame_template:
        args.emit_frame_template.parent.mkdir(parents=True, exist_ok=True)
        args.emit_frame_template.write_text(json.dumps(ProgramFrame.template(), indent=2) + "\n")
        print(f"wrote {args.emit_frame_template} — fill in the four null year fields")
        return 0

    if not args.documents:
        parser.error("give at least one hypothesis.json, or use --emit-frame-template")
    if not args.out:
        parser.error("--out DIR is required")

    if not args.frame:
        # The refusal is the feature. Currency, geography, route, launch year and
        # above all the filing year are analyst decisions, and a guessed filing
        # year is indistinguishable from a sourced one once it is in the file --
        # while setting the protected window, the one number this handoff can
        # actually support.
        print(
            "--frame is required and this adapter will not guess one.\n"
            "Run --emit-frame-template FILE, fill in the four year fields, and pass it back.",
            file=sys.stderr,
        )
        return 2

    try:
        frame = _frame(args.frame)
    except ValidationError as exc:
        fields = ", ".join(str(e["loc"][0]) for e in exc.errors())
        print(
            f"the frame is not usable yet — {fields} still needs an analyst's answer.\n"
            "A template is not a frame; the null fields are the decisions.",
            file=sys.stderr,
        )
        return 2

    emission = emit(load(*args.documents), frame)

    args.out.mkdir(parents=True, exist_ok=True)
    for program in emission.programs:
        path = args.out / f"{program['program_id']}.program.json"
        path.write_text(json.dumps(program, indent=2) + "\n")
        print(f"wrote {path}")

    (args.out / "emission.json").write_text(emission.model_dump_json(indent=2) + "\n")

    # An empty catalogue rather than no catalogue: it makes the ROI model's
    # missing-anchor warning fire, instead of hiding that no price was ever
    # supplied behind a missing file.
    (args.out / "comparables.json").write_text("[]\n")
    print(f"wrote {args.out / 'comparables.json'} — empty: the graph contains no price")

    for skipped in emission.skipped:
        print(f"skipped {skipped.hypothesis_id}: {skipped.reason}")

    if not emission.programs:
        print("no program was emitted — see the skip reasons above")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
