"""The physics baseline must remain the exact preregistered copy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vendor_hash_manifest() -> None:
    """Every vendored module matches its frozen SHA-256 digest."""
    manifest = json.loads((ROOT / "vendor" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == "8219c99088645b7df984752e099a3f873bae773b"
    for relative, expected in manifest["files"].items():
        payload = (ROOT / "vendor" / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest().upper() == expected


def test_readme_quarantine_banner_is_first_line() -> None:
    """The non-production warning is impossible to miss."""
    first = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[0]
    assert first == "EXPERIMENTAL. Not production. Nothing here is validated for loading decisions."


def test_preregistration_hash_is_frozen() -> None:
    """Protocol edits after the root checkpoint fail loudly."""
    payload = (ROOT / "PREREGISTRATION.md").read_bytes()
    assert hashlib.sha256(payload).hexdigest().upper() == (
        "4E50CB8FF5DE827DFC18C0206C56BAA0B127F31F294AAEE2F5737636C1DAC4C6"
    )
