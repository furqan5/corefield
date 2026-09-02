# Copyright 2026 CoreField (Furqan Shakeel)
#
# Licensed under the PolyForm Noncommercial License 1.0.0.
# You may obtain a copy of the License at
#
#     https://polyformproject.org/licenses/noncommercial/1.0.0
#
# Use is permitted for noncommercial purposes only, as that term is defined by
# the License. Commercial use requires a separate licence from the copyright
# holder. This is a source-available licence, not an open-source one.
#
# Versions of this file released before 2026-09-02 were published under the
# Apache License 2.0 and remain available under those terms; that grant is not
# and cannot be revoked.

"""Command-line entry point.

Two commands, both aimed at the moment before any modelling happens:

    corefield --template site_A.csv     hand this file to a utility engineer
    corefield validate site_A.csv       see the gate report for a filled-in file

`validate` deliberately does not fit anything. Looking at the record and
deciding whether it can support an identification is a separate act from
performing one, and collapsing the two is how people end up with parameters
from data that could never have determined them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ingest import AmbientMissingError, load_telemetry, write_template

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corefield",
        description=(
            "Transformer hot-spot estimation from load, ambient and top-oil telemetry. "
            "IEC provenance UNVERIFIED; no field validation exists. See README."
        ),
    )
    parser.add_argument(
        "--template",
        metavar="PATH",
        help="write a blank telemetry CSV, with instructions, to PATH and exit",
    )
    parser.add_argument(
        "--example-rows",
        type=int,
        default=0,
        metavar="N",
        help="include N clearly-marked example rows in the template (default: 0)",
    )

    sub = parser.add_subparsers(dest="command")
    validate = sub.add_parser(
        "validate", help="read a telemetry CSV and print the validation report"
    )
    validate.add_argument("path", help="telemetry CSV to inspect")
    validate.add_argument(
        "--grid-step",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="uniform grid step to resample onto (default: 30)",
    )
    validate.add_argument(
        "--rated-current",
        type=float,
        default=None,
        metavar="AMPS",
        help="nameplate rated current, required if the file carries amperes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.template:
        destination = write_template(args.template, example_rows=args.example_rows)
        print(f"Wrote telemetry template to {destination}")
        print(
            "Hand this to the site engineer. The column notes are in the file header, "
            "including why the ambient column is not optional."
        )
        return 0

    if args.command == "validate":
        try:
            frame = load_telemetry(
                args.path,
                grid_step_s=args.grid_step,
                rated_current_A=args.rated_current,
            )
        except AmbientMissingError as exc:
            print(f"REFUSED: {Path(args.path).name}\n", file=sys.stderr)
            print(exc, file=sys.stderr)
            return 2
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR reading {args.path}: {exc}", file=sys.stderr)
            return 1

        print(frame.report.report())
        if not frame.report.is_fittable:
            print(
                "\nThis record cannot support a four-parameter identification. "
                "The report above states why.",
                file=sys.stderr,
            )
            return 3
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
