from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from corefield_ml_lab.evidence import (
    EvidenceGateError,
    completed_primary_runs,
    require_completed_primary,
    require_e1_passed,
)
from corefield_ml_lab.runstore import begin_primary_run, finish_primary_run


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
    finish_primary_run(run, aggregate)
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


def test_e1_gate_requires_latest_completed_pass(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    _complete(
        root,
        experiment="e1",
        configuration={"field": True},
        seeds=list(range(10)),
        aggregate={"gate": {"overall_status": "pass"}},
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
        json.dumps({"resolved": {"confirmation_required": ["invented"]}}),
        encoding="utf-8",
    )
    with pytest.raises(EvidenceGateError, match="aggregate hash mismatch"):
        completed_primary_runs(root, "e3")
