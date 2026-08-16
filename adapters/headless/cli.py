"""File-in/file-out LABrador integration command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from adapters.headless.runner import HeadlessResponse, run


def _write(output: Path, response: HeadlessResponse) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(response.model_dump_json(indent=2) + "\n")
    if response.status == "COMPLETE":
        return 0
    assert response.error is not None
    print(f"{response.error.reason_code}: {response.error.message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hypgen-run")
    parser.add_argument("--input", type=Path, required=True, help="headless request JSON")
    parser.add_argument("--output", type=Path, required=True, help="result envelope JSON")
    parser.add_argument(
        "--mode",
        choices=("live", "replay"),
        default="live",
        help="live uses the provider-backed Judge; replay is deterministic and never does",
    )
    args = parser.parse_args(argv)

    execution_mode = args.mode.upper()
    output_origin = "LIVE_PROVIDER" if args.mode == "live" else "DETERMINISTIC_REPLAY"
    try:
        payload: Any = json.loads(args.input.read_text())
    except Exception as error:
        return _write(
            args.output,
            HeadlessResponse(
                status="CANNOT_COMPLETE",
                execution_mode=execution_mode,
                output_origin=output_origin,
                error={"reason_code": "INVALID_REQUEST", "message": str(error)},
            ),
        )
    return _write(args.output, run(payload, mode=execution_mode))


if __name__ == "__main__":
    raise SystemExit(main())
