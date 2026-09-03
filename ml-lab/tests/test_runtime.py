from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from corefield_ml_lab import runtime


FIXED_TIME = datetime(2026, 9, 1, 12, 34, 56, tzinfo=timezone.utc)


class _FakeCuda:
    def __init__(self, *, available: bool, count: int) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


def test_cpu_environment_overwrites_accelerator_settings() -> None:
    environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORMS": "cuda,cpu",
    }

    applied = runtime.enforce_cpu_only_environment(environment)

    assert applied == dict(runtime.CPU_ONLY_ENVIRONMENT)
    assert environment["CUDA_VISIBLE_DEVICES"] == "-1"
    assert environment["JAX_PLATFORMS"] == "cpu"
    assert environment["JAX_PLATFORM_NAME"] == "cpu"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"


def test_torch_device_check_accepts_cpu_and_rejects_visible_cuda() -> None:
    cpu_torch = SimpleNamespace(
        __version__="test-cpu",
        cuda=_FakeCuda(available=False, count=0),
    )
    gpu_torch = SimpleNamespace(
        __version__="test-cuda",
        cuda=_FakeCuda(available=True, count=1),
    )

    cpu_status = runtime.require_torch_cpu_only(cpu_torch)

    assert cpu_status.installed
    assert cpu_status.cpu_only
    with pytest.raises(runtime.CpuOnlyViolationError, match="CPU-only execution"):
        runtime.require_torch_cpu_only(gpu_torch)


def test_missing_torch_is_an_allowed_cpu_only_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_optional_import", lambda _: None)

    status = runtime.inspect_torch_device()

    assert not status.installed
    assert status.cpu_only
    assert status.cuda_available is None


def test_seed_is_repeatable_and_recorded_without_optional_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime, "_optional_import", lambda _: None)
    record_path = tmp_path / "records" / "seed.json"

    first = runtime.set_deterministic_seed(
        1729,
        run_id="unit-test",
        record_path=record_path,
        now_utc=FIXED_TIME,
    )
    first_draw = random.random()
    runtime.set_deterministic_seed(1729, run_id="unit-test", now_utc=FIXED_TIME)
    second_draw = random.random()

    assert first_draw == second_draw
    assert first.seed == 1729
    assert not first.numpy_seeded
    assert not first.torch_seeded
    assert json.loads(record_path.read_text(encoding="utf-8")) == first.to_dict()


def test_seed_calls_installed_optional_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    fake_numpy = SimpleNamespace(
        random=SimpleNamespace(seed=lambda value: calls.append(("numpy", value)))
    )
    fake_torch = SimpleNamespace(
        __version__="test",
        cuda=_FakeCuda(available=False, count=0),
        manual_seed=lambda value: calls.append(("torch", value)),
        use_deterministic_algorithms=lambda value: calls.append(("deterministic", value)),
    )

    def fake_import(name: str) -> object | None:
        return {"numpy": fake_numpy, "torch": fake_torch}.get(name)

    monkeypatch.setattr(runtime, "_optional_import", fake_import)
    record = runtime.set_deterministic_seed(7, run_id="backend-test", now_utc=FIXED_TIME)

    assert calls == [("numpy", 7), ("torch", 7), ("deterministic", True)]
    assert record.numpy_seeded
    assert record.torch_seeded
    assert record.torch_deterministic_algorithms


@pytest.mark.parametrize("bad_seed", [-1, 2**32, True, 1.5])
def test_seed_validation_rejects_out_of_contract_values(bad_seed: object) -> None:
    expected = TypeError if isinstance(bad_seed, (bool, float)) else ValueError
    with pytest.raises(expected):
        runtime.set_deterministic_seed(bad_seed, run_id="bad")  # type: ignore[arg-type]


def test_environment_capture_and_json_round_trip(tmp_path: Path) -> None:
    environment = dict(runtime.CPU_ONLY_ENVIRONMENT)
    record = runtime.capture_environment(
        ("package-that-does-not-exist-corefield-ml-lab",),
        environment=environment,
        now_utc=FIXED_TIME,
    )
    output = tmp_path / "environment.json"

    runtime.write_environment_record(output, record)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert record.captured_at_utc == "2026-09-01T12:34:56Z"
    assert record.packages[0].version is None
    assert payload["cpu_environment"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert payload["logical_cpu_count"] == os.cpu_count()
    assert payload["packages"] == [
        {
            "name": "package-that-does-not-exist-corefield-ml-lab",
            "version": None,
        }
    ]


def test_primary_test_sentinel_is_write_once_and_override_is_audited(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sentinel = tmp_path / "state" / "primary-test.json"
    first = runtime.claim_primary_test_access(
        sentinel, run_id="first", seed=11, now_utc=FIXED_TIME
    )
    original = sentinel.read_bytes()

    assert not first.was_override
    with pytest.raises(runtime.PrimaryTestAlreadyClaimedError):
        runtime.claim_primary_test_access(
            sentinel, run_id="second", seed=12, now_utc=FIXED_TIME
        )
    with pytest.raises(ValueError, match="override_reason"):
        runtime.claim_primary_test_access(
            sentinel,
            run_id="second",
            seed=12,
            override=True,
            now_utc=FIXED_TIME,
        )

    with caplog.at_level(logging.WARNING, logger=runtime.__name__):
        claim = runtime.claim_primary_test_access(
            sentinel,
            run_id="second",
            seed=12,
            override=True,
            override_reason="Infrastructure failure: independent audit rerun",
            now_utc=FIXED_TIME,
        )

    assert claim.was_override
    assert sentinel.read_bytes() == original
    assert claim.override_log_path is not None
    entries = claim.override_log_path.read_text(encoding="utf-8").splitlines()
    assert len(entries) == 1
    payload = json.loads(entries[0])
    assert payload["reason"] == "Infrastructure failure: independent audit rerun"
    assert payload["predecessor_run_id"] == "first"
    assert len(payload["predecessor_sentinel_sha256"]) == 64

    third = runtime.claim_primary_test_access(
        sentinel,
        run_id="third",
        seed=12,
        override=True,
        override_reason="Infrastructure failure: second audited rerun",
        override_predecessor_run_id="second",
        now_utc=FIXED_TIME,
    )
    assert third.was_override
    chained = third.override_log_path.read_text(encoding="utf-8").splitlines()
    assert len(chained) == 2
    assert json.loads(chained[1])["predecessor_run_id"] == "second"
    assert "Primary-test access override logged" in caplog.text


def test_sha256_file_streams_exact_bytes(tmp_path: Path) -> None:
    content = bytes(range(256)) * 17
    source = tmp_path / "sample.bin"
    source.write_bytes(content)

    observed = runtime.sha256_file(source, chunk_size_bytes=31)

    assert observed == hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="positive"):
        runtime.sha256_file(source, chunk_size_bytes=0)


def test_process_peak_rss_is_a_positive_byte_count() -> None:
    observed = runtime.process_peak_rss_bytes()

    assert isinstance(observed, int)
    assert observed > 0


def test_memory_gate_is_strictly_below_two_billion_bytes() -> None:
    below = runtime.evaluate_peak_rss_gate(runtime.PEAK_RSS_LIMIT_BYTES - 1)
    equal = runtime.evaluate_peak_rss_gate(runtime.PEAK_RSS_LIMIT_BYTES)

    assert below.passed
    assert below.headroom_bytes == 1
    assert not equal.passed
    assert equal.headroom_bytes == 0
    with pytest.raises(runtime.MemoryLimitExceededError, match="not below"):
        runtime.require_peak_rss_below_limit(runtime.PEAK_RSS_LIMIT_BYTES)
