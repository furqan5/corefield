"""Single CPU-only entry point for the quarantined experiment harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .confirmation import ConfirmationTrigger, run_reserved_confirmation
from .e1 import PrivateFieldConfig
from .e5 import run_e5_primary
from .experiments import run_e1_primary, run_e2_primary, run_e3_primary, run_e4_primary
from .runtime import (
    capture_environment,
    enforce_cpu_only_environment,
    process_peak_rss_bytes,
    require_torch_cpu_only,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m corefield_ml_lab",
        description="CPU-only falsification harness; never for loading decisions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight", help="inspect CPU/runtime state without a primary run")

    e1 = subparsers.add_parser("e1", help="run the write-once E1 reproduction gate")
    e1.add_argument("--private-script", type=Path)
    e1.add_argument("--private-workbook", type=Path)
    e1.add_argument("--override", action="store_true")
    e1.add_argument("--override-reason")

    e3 = subparsers.add_parser("e3", help="run the write-once E3 overload experiment")
    e3.add_argument("--override", action="store_true")
    e3.add_argument("--override-reason")

    e2 = subparsers.add_parser("e2", help="run the write-once scarce-reference sweep")
    e2.add_argument("--override", action="store_true")
    e2.add_argument("--override-reason")

    e4 = subparsers.add_parser("e4", help="run the write-once hull-aware residual test")
    e4.add_argument("--override", action="store_true")
    e4.add_argument("--override-reason")

    e5 = subparsers.add_parser("e5", help="run the write-once conformal-bound test")
    e5.add_argument("--override", action="store_true")
    e5.add_argument("--override-reason")

    confirmation = subparsers.add_parser(
        "confirmation",
        help="run a reserved E2/E3 confirmation only when a primary trigger exists",
    )
    confirmation.add_argument("experiment", choices=("e2", "e3"))
    confirmation.add_argument("--method", default="pinn")
    confirmation.add_argument("--target-load", type=float)
    confirmation.add_argument("--reference-budget", type=int)
    confirmation.add_argument("--prior-run-id", required=True)
    confirmation.add_argument("--override", action="store_true")
    confirmation.add_argument("--override-reason")
    return parser


def _private_config(args: argparse.Namespace) -> PrivateFieldConfig | None:
    supplied = (args.private_script is not None, args.private_workbook is not None)
    if any(supplied) and not all(supplied):
        raise SystemExit("--private-script and --private-workbook must be supplied together")
    if not all(supplied):
        return None
    return PrivateFieldConfig(
        script_path=args.private_script,
        workbook_path=args.private_workbook,
        python_executable=sys.executable,
    )


def _confirmation_trigger(args: argparse.Namespace) -> ConfirmationTrigger:
    if args.experiment == "e3":
        if args.target_load is None:
            raise SystemExit("E3 confirmation requires --target-load")
        if args.reference_budget is not None:
            raise SystemExit("E3 confirmation does not accept --reference-budget")
        return ConfirmationTrigger.e3(args.method, args.target_load)
    if args.reference_budget is None:
        raise SystemExit("E2 confirmation requires --reference-budget")
    if args.target_load is not None:
        raise SystemExit("E2 confirmation does not accept --target-load")
    if args.method != "pinn":
        raise SystemExit("E2 confirmation method must be pinn")
    return ConfirmationTrigger.e2(args.reference_budget)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    enforce_cpu_only_environment()
    root = repository_root()
    if args.command == "preflight":
        torch = require_torch_cpu_only()
        payload = {
            "environment": capture_environment().to_dict(),
            "peak_rss_bytes": process_peak_rss_bytes(),
            "repository": str(root),
            "torch": {
                "cpu_only": torch.cpu_only,
                "cuda_available": torch.cuda_available,
                "version": torch.version,
                "visible_cuda_device_count": torch.visible_cuda_device_count,
            },
        }
    elif args.command == "e1":
        result = run_e1_primary(
            root,
            private_config=_private_config(args),
            override=args.override,
            override_reason=args.override_reason,
        )
        payload = {
            "gate": result["aggregate"]["gate"]["overall_status"],
            "run_id": result["run_id"],
        }
    elif args.command == "e3":
        result = run_e3_primary(
            root,
            override=args.override,
            override_reason=args.override_reason,
        )
        payload = {
            "confirmation_required": result["aggregate"]["resolved"][
                "confirmation_required"
            ],
            "run_id": result["run_id"],
        }
    elif args.command == "e2":
        result = run_e2_primary(
            root,
            override=args.override,
            override_reason=args.override_reason,
        )
        payload = {
            "classification": result["aggregate"]["resolved"][
                "physics_informed_identification_classification"
            ],
            "confirmation_required": result["aggregate"]["resolved"][
                "confirmation_required"
            ],
            "run_id": result["run_id"],
        }
    elif args.command == "e4":
        result = run_e4_primary(
            root,
            override=args.override,
            override_reason=args.override_reason,
        )
        payload = {
            "classification": result["aggregate"]["resolved"]["classification"],
            "run_id": result["run_id"],
        }
    elif args.command == "e5":
        result = run_e5_primary(
            root,
            override=args.override,
            override_reason=args.override_reason,
        )
        payload = {
            "exchangeable_coverage": result["aggregate"]["exchangeable_in_range"]
            ["metrics"]["empirical_coverage"],
            "run_id": result["run_id"],
        }
    elif args.command == "confirmation":
        trigger = _confirmation_trigger(args)
        result = run_reserved_confirmation(
            root,
            trigger=trigger,
            prior_run_id=args.prior_run_id,
            override=args.override,
            override_reason=args.override_reason,
        )
        payload = {
            "confirmation_passed": result["aggregate"]["resolved"][
                "confirmation_passed"
            ],
            "run_id": result["run_id"],
        }
    else:  # pragma: no cover - argparse makes this unreachable.
        raise AssertionError(args.command)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
