from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from corefield_ml_lab.runstore import (
    begin_primary_run,
    canonical_sha256,
    finish_primary_run,
    record_primary_failures,
    require_clean_worktree,
    write_json_exclusive,
)
from corefield_ml_lab.runtime import PrimaryTestAlreadyClaimedError


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


def test_canonical_hash_ignores_mapping_insertion_order() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_exclusive_json_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "one.json"
    write_json_exclusive(target, {"value": 1})
    with pytest.raises(FileExistsError):
        write_json_exclusive(target, {"value": 2})


def test_primary_run_writes_hash_stamped_sentinel_and_manifests(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    run = begin_primary_run(
        root,
        experiment="e3",
        configuration={"width": 16},
        seeds=range(10),
        command=["python", "-m", "corefield_ml_lab", "e3"],
    )
    sentinel = json.loads((root / "run_state" / "e3.primary-access.json").read_text())
    assert run.config_sha256[:16] in sentinel["run_id"]
    start = json.loads((run.run_directory / "manifest.start.json").read_text())
    assert start["protocol_sha256"] == run.protocol_sha256
    final = finish_primary_run(run, {"passed": False})
    assert final["status"] == "completed"
    assert final["memory_gate"]["passed"]
    assert (run.run_directory / "aggregate.json").is_file()


def test_second_primary_claim_refuses_before_new_manifest(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    kwargs = dict(
        experiment="e5",
        configuration={"alpha": 0.05},
        seeds=[61000],
        command=["python", "run.py", "e5"],
    )
    begin_primary_run(root, **kwargs)
    with pytest.raises(PrimaryTestAlreadyClaimedError):
        begin_primary_run(root, **kwargs)


def test_primary_claim_refuses_dirty_worktree_before_writing_sentinel(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    (root / "uncommitted.txt").write_text("not frozen\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean committed Git worktree"):
        begin_primary_run(
            root,
            experiment="e3",
            configuration={"width": 16},
            seeds=range(10),
            command=["python", "-m", "corefield_ml_lab", "e3"],
        )
    assert not (root / "run_state" / "e3.primary-access.json").exists()


def test_clean_worktree_check_detects_and_then_accepts_a_commit(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    require_clean_worktree(root)
    (root / "tracked.txt").write_text("evidence\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked.txt"):
        require_clean_worktree(root)
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "evidence"], cwd=root, check=True)
    require_clean_worktree(root)


def test_same_configuration_override_uses_new_attempt_directory(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    kwargs = dict(
        experiment="e3",
        configuration={"width": 16},
        seeds=list(range(10)),
        command=["python", "-m", "corefield_ml_lab", "e3"],
    )
    first = begin_primary_run(root, **kwargs)
    finish_primary_run(first, {"attempt": 1})
    subprocess.run(["git", "add", "runs", "run_state"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "record first attempt"], cwd=root, check=True)

    second = begin_primary_run(
        root,
        **kwargs,
        override=True,
        override_reason="documented infrastructure failure",
    )
    assert second.run_id == first.run_id + "-attempt02"
    assert second.run_directory != first.run_directory
    assert (second.run_directory / "manifest.start.json").is_file()
    assert (root / "run_state" / "e3.primary-access.json.overrides.jsonl").is_file()
    finish_primary_run(second, {"attempt": 2})


def test_decorated_primary_failure_writes_redacted_final_evidence(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    secret = "C:/private/transformer-field-record.xlsx"

    @record_primary_failures
    def crash() -> None:
        begin_primary_run(
            root,
            experiment="e3",
            configuration={"width": 16},
            seeds=list(range(10)),
            command=["python", "-m", "corefield_ml_lab", "e3"],
        )
        raise RuntimeError(secret)

    with pytest.raises(RuntimeError, match="private"):
        crash()
    run_directory = next((root / "runs" / "e3").iterdir())
    aggregate = json.loads((run_directory / "aggregate.json").read_text())
    final = json.loads((run_directory / "manifest.final.json").read_text())
    assert final["status"] == "failed"
    assert aggregate["failure"]["exception_type"] == "RuntimeError"
    assert aggregate["failure"]["exception_message_persisted"] is False
    assert secret not in json.dumps(aggregate)
