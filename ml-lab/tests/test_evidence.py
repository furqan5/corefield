from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from corefield_ml_lab.evidence import (
    EvidenceGateError,
    completed_primary_runs,
    evidence_provenance,
    require_completed_primary,
    require_e1_passed,
    require_prerequisite_lineage,
)
from corefield_ml_lab.runstore import begin_primary_run, finish_primary_run
from corefield_ml_lab.runtime import sha256_file


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "PREREGISTRATION.md").write_text("frozen\n", encoding="utf-8")
    (root / "vendor").mkdir()
    (root / "vendor" / "manifest.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "lab@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Lab Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=root, check=True)
    return root


def _complete(
    root: Path,
    *,
    experiment: str,
    configuration: dict[str, object],
    seeds: list[int],
    aggregate: dict[str, object],
) -> None:
    run = begin_primary_run(
        root,
        experiment=experiment,
        configuration=configuration,
        seeds=seeds,
        command=["python", "-m", "corefield_ml_lab", experiment],
    )
    payload = {
        "configuration": configuration,
        "schema_version": 1,
        **aggregate,
    }
    finish_primary_run(run, payload)
    subprocess.run(["git", "add", "runs", "run_state"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", f"record {experiment}"], cwd=root, check=True)


def test_completed_evidence_is_hash_verified_and_exactly_matchable(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    config = {"width": 16, "loads": [1.0, 1.6]}
    _complete(
        root,
        experiment="e3",
        configuration=config,
        seeds=list(range(10)),
        aggregate={"resolved": {"confirmation_required": []}},
    )
    records = completed_primary_runs(root, "e3")
    assert len(records) == 1
    assert records[0].configuration == config
    assert require_completed_primary(
        root,
        "e3",
        required_configuration=config,
        required_seeds=range(10),
    ).run_id == records[0].run_id
    with pytest.raises(EvidenceGateError, match="none match"):
        require_completed_primary(
            root,
            "e3",
            required_configuration={"width": 32},
            required_seeds=range(10),
        )


def test_completed_third_attempt_requires_an_unbroken_override_chain(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    kwargs = dict(
        experiment="e3",
        configuration={"width": 16},
        seeds=list(range(10)),
        command=["python", "-m", "corefield_ml_lab", "e3"],
    )
    first = begin_primary_run(root, **kwargs)
    finish_primary_run(
        first,
        {"configuration": {"width": 16}, "schema_version": 1},
        status="failed",
    )
    subprocess.run(["git", "add", "runs", "run_state"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "failed first"], cwd=root, check=True)

    second = begin_primary_run(
        root,
        **kwargs,
        override=True,
        override_reason="Infrastructure failure: first retry",
    )
    finish_primary_run(
        second,
        {"configuration": {"width": 16}, "schema_version": 1},
        status="failed",
    )
    subprocess.run(["git", "add", "runs", "run_state"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "failed second"], cwd=root, check=True)

    third = begin_primary_run(
        root,
        **kwargs,
        override=True,
        override_reason="Infrastructure failure: second retry",
    )
    finish_primary_run(
        third,
        {"configuration": {"width": 16}, "schema_version": 1},
    )
    subprocess.run(["git", "add", "runs", "run_state"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "completed third"], cwd=root, check=True)

    records = completed_primary_runs(root, "e3")
    assert [record.run_id for record in records] == [third.run_id]
    assert third.run_id.endswith("-attempt03")


def test_e5_evidence_requires_consistent_process_tree_memory_bound(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    _complete(
        root,
        experiment="e5",
        configuration={"alpha": 0.05},
        seeds=[61_000],
        aggregate={"result": "fixture"},
    )
    run_directory = next((root / "runs" / "e5").iterdir())
    final_path = run_directory / "manifest.final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["memory_measurement"]["conservative_process_tree_upper_bound_bytes"] += 1
    final_path.write_text(json.dumps(final), encoding="utf-8")

    with pytest.raises(EvidenceGateError, match="process-tree memory bound"):
        completed_primary_runs(root, "e5")


def test_e1_gate_requires_latest_completed_pass(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    seeds = list(range(1_000, 1_010))
    configuration = {
        "day": "C",
        "private_field_access_configured": True,
        "private_output_policy": "aggregate metrics only; no stdout or rows retained",
        "seeds": seeds,
    }
    _complete(
        root,
        experiment="e1",
        configuration=configuration,
        seeds=seeds,
        aggregate={
            "field": {"hotspot": {"rmse_K": 1.55}},
            "gate": {
                "field": {"status": "pass"},
                "overall_status": "pass",
                "synthetic": {"status": "pass"},
            },
        },
    )
    assert require_e1_passed(root).experiment == "e1"


def test_tampered_aggregate_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _complete(
        root,
        experiment="e3",
        configuration={"width": 16},
        seeds=list(range(10)),
        aggregate={"resolved": {"confirmation_required": []}},
    )
    aggregate_path = next((root / "runs" / "e3").glob("*/aggregate.json"))
    aggregate_path.write_text(
        json.dumps(
            {
                "configuration": {"width": 16},
                "resolved": {"confirmation_required": ["invented"]},
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvidenceGateError, match="aggregate hash mismatch"):
        completed_primary_runs(root, "e3")


def test_required_configuration_does_not_alias_boolean_and_integer(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _complete(
        root,
        experiment="e3",
        configuration={"flag": 1},
        seeds=list(range(10)),
        aggregate={"resolved": {"confirmation_required": []}},
    )
    with pytest.raises(EvidenceGateError, match="none match"):
        require_completed_primary(
            root,
            "e3",
            required_configuration={"flag": True},
            required_seeds=range(10),
        )


def test_missing_access_sentinel_is_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _complete(
        root,
        experiment="e3",
        configuration={"width": 16},
        seeds=list(range(10)),
        aggregate={"resolved": {"confirmation_required": []}},
    )
    (root / "run_state" / "e3.primary-access.json").unlink()
    with pytest.raises(EvidenceGateError, match="cannot read evidence JSON"):
        completed_primary_runs(root, "e3")


def test_internal_configuration_and_memory_forgery_are_rejected(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _complete(
        root,
        experiment="e3",
        configuration={"width": 16},
        seeds=list(range(10)),
        aggregate={"resolved": {"confirmation_required": []}},
    )
    run_directory = next((root / "runs" / "e3").iterdir())
    aggregate_path = run_directory / "aggregate.json"
    final_path = run_directory / "manifest.final.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["configuration"] = {"width": 32}
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    final = json.loads(final_path.read_text())
    final["aggregate_sha256"] = sha256_file(aggregate_path)
    final_path.write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(EvidenceGateError, match="configuration mismatch"):
        completed_primary_runs(root, "e3")

    aggregate["configuration"] = {"width": 16}
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    final["aggregate_sha256"] = sha256_file(aggregate_path)
    final["memory_gate"] = {
        "headroom_bytes": 0,
        "limit_bytes": 2_000_000_000,
        "passed": True,
        "peak_rss_bytes": 2_000_000_000,
    }
    final_path.write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(EvidenceGateError, match="memory gate arithmetic"):
        completed_primary_runs(root, "e3")


def test_prerequisite_lineage_is_exact(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    predecessor_config = {"width": 16}
    _complete(
        root,
        experiment="e3",
        configuration=predecessor_config,
        seeds=list(range(10)),
        aggregate={"resolved": {"confirmation_required": []}},
    )
    predecessor = require_completed_primary(root, "e3")
    dependent_config = {"alpha": 0.05}
    run = begin_primary_run(
        root,
        experiment="e5",
        configuration=dependent_config,
        seeds=[61000],
        command=["python", "-m", "corefield_ml_lab", "e5"],
        prerequisites=[evidence_provenance(predecessor)],
    )
    finish_primary_run(
        run,
        {"configuration": dependent_config, "schema_version": 1},
    )
    dependent = require_completed_primary(root, "e5")
    require_prerequisite_lineage(dependent, [predecessor])
    fake = replace(predecessor, run_id="e3-fabricated")
    with pytest.raises(EvidenceGateError, match="lineage"):
        require_prerequisite_lineage(dependent, [fake])
