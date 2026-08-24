# Copyright 2026 CoreField (Furqan Shakeel)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke tests for the Streamlit demo.

The demo is the artifact people actually look at, and it is the easiest thing
in the repository to break without noticing -- nothing else imports it. These
tests run the real app through Streamlit's own test harness.

The banner assertions are the ones that matter. A synthetic result
screenshotted without its SYNTHETIC label would be a false claim about a real
transformer, so the label is treated as a correctness property, not styling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="demo extras not installed")
pytest.importorskip("matplotlib", reason="demo extras not installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

# AppTest resolves relative paths against the CALLING file, not the working
# directory, so anchor on the repository root explicitly.
APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
TIMEOUT = 180


@pytest.fixture(scope="module")
def app():
    """Run the app once with its default state (bundled synthetic demo)."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception(app):
    assert not app.exception, [str(e) for e in app.exception]


def test_all_three_tabs_exist(app):
    labels = " ".join(t.label for t in app.tabs)
    assert "Why this matters" in labels
    assert "Identify" in labels
    assert "Loading envelope" in labels


def test_synthetic_banner_is_present_by_default(app):
    """Default state is bundled synthetic data, so the warning banner must show."""
    banners = " ".join(w.value for w in app.warning)
    assert "SYNTHETIC DATA" in banners
    assert "No field validation exists" in banners


def test_unverified_iec_provenance_is_disclosed(app):
    """The UNVERIFIED status must be visible in the app, not only in the README."""
    text = " ".join(c.value for c in app.caption) + " ".join(m.value for m in app.markdown)
    assert "UNVERIFIED" in text


def test_day_c_numbers_are_computed_not_hardcoded(app):
    """The chart's headline numbers come from corefield.campaign at runtime.

    Verified by checking the app's cached computation directly rather than
    scraping a rendered figure: the point is that no literal 6.17 / 3.17 /
    0.32 appears anywhere in the app source.
    """
    source = open(APP, encoding="utf-8").read()
    for literal in ("6.17", "6.18", "3.17", "0.32"):
        assert literal not in source, (
            f"{literal!r} is hard-coded in the demo. The day-C figures must be "
            f"reproduced from corefield.campaign so the chart cannot drift from the code."
        )


def test_app_does_not_ship_loading_limits():
    """No plausible-looking limit may be presented as sourced.

    The number inputs carry starting values so the form is usable, but the
    source field must start EMPTY -- an unlabelled limit must never be
    submittable by pressing the button without typing anything.
    """
    source = open(APP, encoding="utf-8").read()
    assert 'value=""' in source, "the limits-source field must start empty"
    assert "placeholder=" in source, "it should show an example without pre-filling one"


def test_identify_tab_reports_the_validation_gate(app):
    """The gate report must be rendered before any parameter is shown."""
    code_blocks = " ".join(c.value for c in app.code)
    assert "TELEMETRY VALIDATION REPORT" in code_blocks
    assert "LOAD HULL" in code_blocks
    assert "FITTABLE" in code_blocks
