from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import corefield_ml_lab.confirmation as confirmation
from corefield_ml_lab.confirmation import (
    ConfirmationTrigger,
    RESERVED_CONFIRMATION_SEEDS,
    RESERVED_HIDDEN_WIDTHS,
    VerifiedPrimaryTrigger,
    aggregate_confirmation_rows,
    confirmation_configuration,
    run_reserved_confirmation,
    recorded_confirmation_command,
    verify_completed_primary_trigger,
)
from corefield_ml_lab.experiments import (
    E2_SEEDS,
    E3_SEEDS,
    e2_configuration,
    e3_configuration,
)


def _metric_rows(
    trigger: ConfirmationTrigger,
    *,
    method_rmse: dict[int, float],
    method_peak: dict[int, float],
    nls_rmse: float = 1.0,
    nls_peak: float = 0.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for width in RESERVED_HIDDEN_WIDTHS:
        for seed in RESERVED_CONFIRMATION_SEEDS:
            for method, rmse, peak in (
                ("nls", nls_rmse, nls_peak),
                (trigger.method, method_rmse[width], method_peak[width]),
            ):
                rows.append(
                    {
                        "hidden_width": width,
                        "mae_K": rmse,
                        "mean_signed_error_K": peak,
                        "method": method,
                        "rmse_K": rmse,
                        "seed": seed,
                        "signed_peak_error_K": peak,
                    }
                )
    return rows


def _parameter_rows(*, finite: bool = True) -> list[dict[str, object]]:
    return [
        {
            "finite_interior": finite,
            "hidden_width": width,
            "method": "pinn",
            "seed": seed,
        }
        for width in RESERVED_HIDDEN_WIDTHS
        for seed in RESERVED_CONFIRMATION_SEEDS
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fake_prior_run(
    root: Path,
    trigger: ConfirmationTrigger,
    *,
    include_trigger: bool = True,
) -> str:
    root.mkdir(parents=True, exist_ok=True)
    preregistration = root / "PREREGISTRATION.md"
    preregistration.write_text("frozen protocol\n", encoding="utf-8")
    protocol_hash = _sha256(preregistration)
    vendor_manifest = root / "vendor" / "manifest.json"
    _write_json(vendor_manifest, {})
    vendor_hash = _sha256(vendor_manifest)
    if trigger.experiment == "e3":
        required: dict[str, object] = {
            "method": trigger.method,
            "target_load_pu": trigger.target_load_pu,
            "reserved_seeds": list(RESERVED_CONFIRMATION_SEEDS),
            "widths": list(RESERVED_HIDDEN_WIDTHS),
        }
    else:
        required = {
            "experiment": "e2",
            "method": "pinn",
            "reference_budget": trigger.reference_budget,
            "reserved_seeds": list(RESERVED_CONFIRMATION_SEEDS),
            "widths": list(RESERVED_HIDDEN_WIDTHS),
        }
    configuration = (
        e3_configuration() if trigger.experiment == "e3" else e2_configuration()
    )
    primary_seeds = E3_SEEDS if trigger.experiment == "e3" else E2_SEEDS
    config_payload = {
        "configuration": configuration,
        "experiment": trigger.experiment,
        "protocol_sha256": protocol_hash,
        "schema_version": 1,
        "seeds": list(primary_seeds),
        "vendor_manifest_sha256": vendor_hash,
    }
    config_hash = _canonical_sha256(config_payload)
    run_id = f"{trigger.experiment}-{config_hash[:16]}"
    directory = root / "runs" / trigger.experiment / run_id
    aggregate = {
        "configuration": configuration,
        "resolved": {"confirmation_required": [required] if include_trigger else []}
    }
    aggregate_path = directory / "aggregate.json"
    _write_json(aggregate_path, aggregate)
    start = {
        "config_payload": config_payload,
        "config_sha256": config_hash,
        "experiment": trigger.experiment,
        "protocol_sha256": protocol_hash,
        "run_id": run_id,
        "vendor_manifest_sha256": vendor_hash,
    }
    final = {
        "aggregate_sha256": _sha256(aggregate_path),
        "config_sha256": config_hash,
        "experiment": trigger.experiment,
        "memory_gate": {"passed": True},
        "protocol_sha256": protocol_hash,
        "run_id": run_id,
        "status": "completed",
    }
    _write_json(directory / "manifest.start.json", start)
    _write_json(directory / "manifest.final.json", final)
    return run_id


def test_trigger_restricts_experiment_method_load_and_budget() -> None:
    assert ConfirmationTrigger.e3("plain_nn", 1.30).sentinel_experiment == (
        "e3_confirmation_plain_nn_130"
    )
    assert ConfirmationTrigger.e2(3).sentinel_experiment == (
        "e2_confirmation_pinn_n3"
    )
    with pytest.raises(ValueError, match="load"):
        ConfirmationTrigger.e3("pinn", 1.15)
    with pytest.raises(ValueError, match="method"):
        ConfirmationTrigger.e3("generic_iec", 1.30)
    with pytest.raises(ValueError, match="budget"):
        ConfirmationTrigger.e2(5)


def test_configuration_freezes_reserved_seeds_widths_and_training() -> None:
    trigger = ConfirmationTrigger.e3("pinn", 1.60)
    source = VerifiedPrimaryTrigger(
        "e3", "e3-source", "a" * 64, "b" * 64, "c" * 64, "source/final.json"
    )
    config = confirmation_configuration(trigger, source)
    assert tuple(config["reserved_seeds"]) == RESERVED_CONFIRMATION_SEEDS
    assert tuple(config["hidden_widths"]) == RESERVED_HIDDEN_WIDTHS
    assert config["neural_training"]["max_epochs"] == 2_000
    assert config["neural_training"]["patience"] == 150
    assert config["reference_budget"] == 20
    assert config["physics_mode"] == "structural_mismatch"
    assert config["exact_reference_features"] is True
    assert config["measurement_noise"]["top_oil_actual_sample_interval_s"] == 300.0


def test_recorded_confirmation_command_contains_exact_trigger_and_override() -> None:
    trigger = ConfirmationTrigger.e3("pinn", 1.30)
    command = recorded_confirmation_command(
        trigger,
        prior_run_id="e3-source",
        override=True,
        override_reason="documented confirmation infrastructure failure",
    )
    assert command[3:5] == ["confirmation", "e3"]
    assert command[command.index("--target-load") + 1] == "1.30"
    assert command[command.index("--prior-run-id") + 1] == "e3-source"
    assert command[-2:] == [
        "--override-reason",
        "documented confirmation infrastructure failure",
    ]


def test_e3_confirmation_requires_both_widths_safe_and_one_accuracy_win() -> None:
    trigger = ConfirmationTrigger.e3("pinn", 1.30)
    rows = _metric_rows(
        trigger,
        method_rmse={8: 0.5, 32: 1.1},
        method_peak={8: -0.10, 32: -0.10},
    )
    result = aggregate_confirmation_rows(trigger, rows, [])
    assert result["rule"]["both_widths_preserve_safety"]
    assert result["rule"]["at_least_one_width_preserves_paired_rmse_win"]
    assert result["confirmation_passed"]

    unsafe = _metric_rows(
        trigger,
        method_rmse={8: 0.5, 32: 0.6},
        method_peak={8: -0.10, 32: -0.1000001},
    )
    result = aggregate_confirmation_rows(trigger, unsafe, [])
    assert not result["rule"]["both_widths_preserve_safety"]
    assert not result["confirmation_passed"]


def test_e2_n_ge_4_uses_strict_no_worse_worst_peak() -> None:
    trigger = ConfirmationTrigger.e2(4)
    rows = _metric_rows(
        trigger,
        method_rmse={8: 0.5, 32: 0.6},
        method_peak={8: 0.0, 32: -0.000001},
        nls_peak=0.0,
    )
    result = aggregate_confirmation_rows(trigger, rows, _parameter_rows())
    assert result["width_results"]["8"]["strict_no_worse_worst_signed_peak"]
    assert not result["width_results"]["32"][
        "strict_no_worse_worst_signed_peak"
    ]
    assert not result["confirmation_passed"]


def test_e2_n3_requires_both_widths_all_finite_interior_and_each_rmse_below_2() -> None:
    trigger = ConfirmationTrigger.e2(3)
    rows = _metric_rows(
        trigger,
        method_rmse={8: 1.9, 32: 1.99},
        method_peak={8: -100.0, 32: -100.0},
    )
    result = aggregate_confirmation_rows(trigger, rows, _parameter_rows())
    assert result["rule"]["both_widths_pass_n3_screen"]
    assert result["confirmation_passed"]
    assert result["preregistered_consequence"] == "investigate further"

    at_boundary = _metric_rows(
        trigger,
        method_rmse={8: 1.9, 32: 2.0},
        method_peak={8: 0.0, 32: 0.0},
    )
    assert not aggregate_confirmation_rows(
        trigger, at_boundary, _parameter_rows()
    )["confirmation_passed"]


def test_aggregation_refuses_nonreserved_seed_or_width_sets() -> None:
    trigger = ConfirmationTrigger.e3("plain_nn", 1.60)
    with pytest.raises(ValueError, match="exactly seeds"):
        aggregate_confirmation_rows(
            trigger,
            [],
            [],
            seeds=tuple(range(10)),
        )
    with pytest.raises(ValueError, match="exactly hidden widths"):
        aggregate_confirmation_rows(trigger, [], [], widths=(8, 16))


def test_verifier_accepts_only_completed_hash_checked_exact_trigger(
    tmp_path: Path,
) -> None:
    trigger = ConfirmationTrigger.e3("pinn", 1.30)
    run_id = _fake_prior_run(tmp_path, trigger)
    verified = verify_completed_primary_trigger(
        tmp_path, trigger=trigger, prior_run_id=run_id
    )
    assert verified.run_id == run_id
    assert verified.experiment == "e3"
    assert verified.aggregate_sha256 == _sha256(
        tmp_path / "runs" / "e3" / run_id / "aggregate.json"
    )

    aggregate_path = tmp_path / "runs" / "e3" / run_id / "aggregate.json"
    aggregate_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash"):
        verify_completed_primary_trigger(
            tmp_path, trigger=trigger, prior_run_id=run_id
        )


def test_verifier_does_not_claim_an_unlisted_or_wrong_reserved_trigger(
    tmp_path: Path,
) -> None:
    trigger = ConfirmationTrigger.e2(6)
    run_id = _fake_prior_run(tmp_path, trigger, include_trigger=False)
    with pytest.raises(RuntimeError, match="exactly one"):
        verify_completed_primary_trigger(
            tmp_path, trigger=trigger, prior_run_id=run_id
        )

    other = tmp_path / "other"
    run_id = _fake_prior_run(other, trigger)
    aggregate_path = other / "runs" / "e2" / run_id / "aggregate.json"
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    payload["resolved"]["confirmation_required"][0]["widths"] = [8.9, 32]
    _write_json(aggregate_path, payload)
    final_path = other / "runs" / "e2" / run_id / "manifest.final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["aggregate_sha256"] = _sha256(aggregate_path)
    _write_json(final_path, final)
    with pytest.raises(RuntimeError, match="exactly one"):
        verify_completed_primary_trigger(
            other, trigger=trigger, prior_run_id=run_id
        )


def test_runner_verifies_then_claims_before_post_claim_executor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    trigger = ConfirmationTrigger.e3("plain_nn", 1.60)
    events: list[str] = []
    source = VerifiedPrimaryTrigger(
        "e3", "e3-source", "a" * 64, "b" * 64, "c" * 64, "final.json"
    )

    monkeypatch.setattr(confirmation, "enforce_cpu_only_environment", lambda: events.append("cpu"))
    monkeypatch.setattr(
        confirmation,
        "require_torch_cpu_only",
        lambda: SimpleNamespace(
            cpu_only=True,
            cuda_available=False,
            version="test",
            visible_cuda_device_count=0,
        ),
    )

    def verify(*args: object, **kwargs: object) -> VerifiedPrimaryTrigger:
        events.append("verify")
        return source

    def begin(*args: object, **kwargs: object) -> SimpleNamespace:
        events.append("claim")
        assert kwargs["seeds"] == RESERVED_CONFIRMATION_SEEDS
        assert kwargs["experiment"] == trigger.sentinel_experiment
        return SimpleNamespace(run_id="confirmation-test")

    def execute(received: ConfirmationTrigger) -> dict[str, object]:
        events.append("hidden_truth_executor")
        assert received == trigger
        assert "claim" in events
        return {
            "availability_rows": [],
            "parameter_rows": [],
            "resolved": {"confirmation_passed": False},
            "rows": [],
            "training": [],
        }

    def finish(run: object, aggregate: object) -> dict[str, object]:
        events.append("finish")
        return {"status": "completed"}

    monkeypatch.setattr(confirmation, "verify_completed_primary_trigger", verify)
    monkeypatch.setattr(confirmation, "begin_primary_run", begin)
    monkeypatch.setattr(confirmation, "_execute_confirmation_after_claim", execute)
    monkeypatch.setattr(confirmation, "finish_primary_run", finish)

    result = run_reserved_confirmation(
        tmp_path,
        trigger=trigger,
        prior_run_id="e3-source",
    )
    assert events == ["cpu", "verify", "claim", "hidden_truth_executor", "finish"]
    assert result["run_id"] == "confirmation-test"
