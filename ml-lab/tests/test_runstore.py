from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

import corefield_ml_lab.runstore as runstore_module
from corefield_ml_lab.runstore import (
    begin_primary_run,
    canonical_sha256,
    finish_primary_run,
    record_primary_failures,
    require_active_primary_run,
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


def test_exclusive_json_serializes_before_claiming_path(tmp_path: Path) -> None:
    target = tmp_path / "invalid.json"
    with pytest.raises(TypeError):
        write_json_exclusive(target, {"value": object()})
    assert not target.exists()


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
    require_active_primary_run(run, experiment="e3")
    final = finish_primary_run(run, {"passed": False})
    assert final["status"] == "completed"
    assert final["memory_gate"]["passed"]
    measurement = final["memory_measurement"]
    assert measurement["process_tree_equivalent"] is True
    assert measurement["child_peak_observation_complete"] is True
    assert measurement["conservative_process_tree_upper_bound_bytes"] == (
        measurement["current_process_peak_rss_bytes"]
        + measurement["maximum_observed_direct_child_peak_rss_bytes"]
    )
    assert final["memory_gate"]["peak_rss_bytes"] == measurement[
        "conservative_process_tree_upper_bound_bytes"
    ]
    assert (run.run_directory / "aggregate.json").is_file()
    with pytest.raises(RuntimeError, match="active post-sentinel"):
        require_active_primary_run(run, experiment="e3")


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
    finish_primary_run(first, {"attempt": 1}, status="failed")
    subprocess.run(["git", "add", "runs", "run_state"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "record first attempt"], cwd=root, check=True)

    second = begin_primary_run(
        root,
        **kwargs,
        override=True,
        override_reason="Infrastructure failure: interrupted first attempt",
    )
    assert second.run_id == first.run_id + "-attempt02"
    assert second.run_directory != first.run_directory
    assert (second.run_directory / "manifest.start.json").is_file()
    assert (root / "run_state" / "e3.primary-access.json.overrides.jsonl").is_file()
    finish_primary_run(second, {"attempt": 2})


def test_completed_same_configuration_cannot_be_overridden(tmp_path: Path) -> None:
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
    subprocess.run(["git", "commit", "-qm", "record completed run"], cwd=root, check=True)
    with pytest.raises(RuntimeError, match="completed, memory-passing"):
        begin_primary_run(
            root,
            **kwargs,
            override=True,
            override_reason="Infrastructure failure: unsupported repeat request",
        )


def test_override_cannot_change_the_original_configuration(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    first = begin_primary_run(
        root,
        experiment="e3",
        configuration={"width": 16},
        seeds=list(range(10)),
        command=["python", "-m", "corefield_ml_lab", "e3"],
    )
    finish_primary_run(first, {"attempt": 1}, status="failed")
    subprocess.run(["git", "add", "runs", "run_state"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "record failed run"], cwd=root, check=True)

    with pytest.raises(RuntimeError, match="exact configuration"):
        begin_primary_run(
            root,
            experiment="e3",
            configuration={"width": 32},
            seeds=list(range(10)),
            command=["python", "-m", "corefield_ml_lab", "e3"],
            override=True,
            override_reason="Infrastructure failure: changed configuration attempt",
        )


def test_completed_later_attempt_cannot_be_overridden(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    kwargs = dict(
        experiment="e3",
        configuration={"width": 16},
        seeds=list(range(10)),
        command=["python", "-m", "corefield_ml_lab", "e3"],
    )
    first = begin_primary_run(root, **kwargs)
    finish_primary_run(first, {"attempt": 1}, status="failed")
    subprocess.run(["git", "add", "runs", "run_state"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "record first attempt"], cwd=root, check=True)
    second = begin_primary_run(
        root,
        **kwargs,
        override=True,
        override_reason="Infrastructure failure: retry first failure",
    )
    finish_primary_run(second, {"attempt": 2})
    subprocess.run(["git", "add", "runs", "run_state"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "record completed retry"], cwd=root, check=True)

    with pytest.raises(RuntimeError, match="completed, memory-passing"):
        begin_primary_run(
            root,
            **kwargs,
            override=True,
            override_reason="Infrastructure failure: forbidden third attempt",
        )


def test_override_requires_existing_sentinel(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    with pytest.raises(RuntimeError, match="existing access sentinel"):
        begin_primary_run(
            root,
            experiment="e3",
            configuration={"width": 16},
            seeds=list(range(10)),
            command=["python", "-m", "corefield_ml_lab", "e3"],
            override=True,
            override_reason="Infrastructure failure: missing predecessor",
        )


def test_memory_gate_failure_is_terminally_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repository(tmp_path)
    run = begin_primary_run(
        root,
        experiment="e5",
        configuration={"alpha": 0.05},
        seeds=[61000],
        command=["python", "-m", "corefield_ml_lab", "e5"],
    )
    monkeypatch.setattr(
        runstore_module,
        "process_peak_rss_bytes",
        lambda: 2_000_000_000,
    )
    with pytest.raises(MemoryError):
        finish_primary_run(run, {"result": "complete"})
    final = json.loads((run.run_directory / "manifest.final.json").read_text())
    assert final["status"] == "failed"
    assert final["memory_gate"]["passed"] is False


def test_failure_after_existing_aggregate_gets_emergency_terminal_marker(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)

    @record_primary_failures
    def crash_after_aggregate() -> None:
        run = begin_primary_run(
            root,
            experiment="e3",
            configuration={"width": 16},
            seeds=list(range(10)),
            command=["python", "-m", "corefield_ml_lab", "e3"],
        )
        write_json_exclusive(run.run_directory / "aggregate.json", {"partial": True})
        raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        crash_after_aggregate()
    run_directory = next((root / "runs" / "e3").iterdir())
    emergency = json.loads((run_directory / "manifest.failure.json").read_text())
    assert emergency["status"] == "failed"
    assert emergency["failure"]["exception_message_persisted"] is False


def test_every_primary_seed_requires_an_exact_unsigned_integer(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    for invalid in ([0, 1.5], [0, True], [0, -1], [0, 2**32]):
        with pytest.raises(ValueError, match="every seed"):
            begin_primary_run(
                root,
                experiment="e3",
                configuration={"width": 16},
                seeds=invalid,
                command=["python", "-m", "corefield_ml_lab", "e3"],
            )


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
