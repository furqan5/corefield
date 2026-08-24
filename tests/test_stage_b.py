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

"""Stage B regression: the structural-mismatch gate that selected Model C.

Pre-registered gate: RMSE <= 2 K AND |peak error| <= 2 K, against the TRUE
hot-spot trajectory, on the fitting day AND on an unseen day.

The peak criterion is separate from RMSE deliberately. A model can flatter
its RMSE while missing the one moment an operator cares about -- Model B on
day B is exactly that case, with RMSE improving to 1.73 K while its worst
absolute error GROWS to 5.4 K at the new event times.
"""

from __future__ import annotations

import pytest

from conftest import assert_reproduces

from corefield.campaign import GATE_PEAK_K, GATE_RMSE_K


# --------------------------------------------------------------------------
# Day-A gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, rmse, peak, event_rmse",
    [
        ("A", 2.59, +4.03, 2.67),
        ("B", 1.82, +2.24, 2.65),
        ("C", 0.10, -0.02, 0.16),
    ],
)
def test_day_a_gate_table(day_a_comparison, model, rmse, peak, event_rmse):
    """The published day-A gate table, reproduced fit for fit."""
    assert_reproduces(day_a_comparison.mean(model, "rmse_K"), rmse, f"{model} day-A RMSE")
    assert_reproduces(
        day_a_comparison.mean(model, "peak_error_K"), peak, f"{model} day-A peak error"
    )
    assert_reproduces(
        day_a_comparison.mean(model, "event_rmse_K"), event_rmse, f"{model} day-A event RMSE"
    )


@pytest.mark.parametrize("model, worst", [("A", 5.32), ("B", 4.53)])
def test_day_a_worst_absolute_error(day_a_comparison, model, worst):
    """Worst absolute error on day A for the two falsified models."""
    assert_reproduces(day_a_comparison.mean(model, "max_abs_K"), worst, f"{model} day-A worst")


def test_model_c_day_a_worst_absolute_error(day_a_comparison):
    """Model C's day-A worst absolute error.

    PINNED AT 0.26, NOT the 0.21 that appears in methods v3/v4 section 6.2
    and in the cell-30 markdown table.

    This is the one published number in the entire campaign that does not
    reproduce, and the evidence says the published value was never computed
    by the notebook at all: the Stage-B gate aggregation (v4 cell 31) builds
    its table from rmse / rmse_wc / peak_err / ev_rmse and has NO maxerr
    column for Model C. The 0.21 was hand-entered into a markdown table.

    Everything else about Model C reproduces exactly -- RMSE 0.10, peak
    -0.02, event RMSE 0.16, all four parameter errors, the noiseless case
    at exactly 0.00 %, and the n=5 / n=9 calibration floors at 13.34 % /
    2.26 %. Fits that agree to the digit on nine quantities cannot disagree
    by 24 % on the tenth; the discrepancy is in the transcription, not the
    code. Per-seed values reproduce as
    [0.30, 0.25, 0.09, 0.15, 0.20, 0.22, 0.46, 0.31, 0.34, 0.29].

    See AUDIT.md and REPRODUCTION.md. Do not "fix" this by relaxing the
    tolerance to admit 0.21.
    """
    assert_reproduces(day_a_comparison.mean("C", "max_abs_K"), 0.26, "C day-A worst")


@pytest.mark.parametrize("model", ["A", "B"])
def test_single_exponential_models_fail_the_gate(day_a_comparison, model):
    """A and B FAIL day A. This is a result -- assert it, do not skip it.

    Both fail in the OVER-prediction direction, which is the commercially
    damaging one: a monitor reading high triggers derating that was never
    needed.
    """
    peak = day_a_comparison.mean(model, "peak_error_K")
    rmse = day_a_comparison.mean(model, "rmse_K")
    assert not (rmse <= GATE_RMSE_K and abs(peak) <= GATE_PEAK_K), (
        f"Model {model} unexpectedly PASSED the day-A gate "
        f"(RMSE {rmse}, peak {peak}). The published campaign records it as FAIL."
    )
    assert peak > 0.0, f"Model {model} must fail HIGH (false-alarm direction), got {peak}"


def test_model_c_passes_the_gate(day_a_comparison):
    """Model C passes both gate criteria on the fitting day."""
    rmse = day_a_comparison.mean("C", "rmse_K")
    peak = day_a_comparison.mean("C", "peak_error_K")
    assert rmse <= GATE_RMSE_K and abs(peak) <= GATE_PEAK_K


# --------------------------------------------------------------------------
# Day-B transfer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, rmse, worst, peak",
    [
        ("A", 2.10, 4.10, +3.14),
        ("B", 1.73, 5.40, +1.90),
        ("C", 0.11, 0.30, -0.02),
    ],
)
def test_day_b_transfer_table(day_b_comparison, model, rmse, worst, peak):
    """The published day-B transfer table: fitted on day A, scored on day B."""
    assert_reproduces(day_b_comparison.mean(model, "rmse_K"), rmse, f"{model} day-B RMSE")
    assert_reproduces(day_b_comparison.mean(model, "max_abs_K"), worst, f"{model} day-B worst")
    assert_reproduces(day_b_comparison.mean(model, "peak_error_K"), peak, f"{model} day-B peak")


def test_model_b_worst_case_grows_on_day_b(day_a_comparison, day_b_comparison):
    """B's RMSE improves on day B while its worst case gets worse.

    1.82 -> 1.73 K RMSE, but 4.53 -> 5.40 K worst absolute error. This is
    precisely why worst case is a gate metric and RMSE alone is not.
    """
    assert day_b_comparison.mean("B", "rmse_K") < day_a_comparison.mean("B", "rmse_K")
    assert day_b_comparison.mean("B", "max_abs_K") > day_a_comparison.mean("B", "max_abs_K")


def test_model_ranking_is_preserved_on_the_unseen_day(day_b_comparison):
    """C < B < A on RMSE, on a day none of them were fitted to."""
    rmse = {m: day_b_comparison.mean(m, "rmse_K") for m in ("A", "B", "C")}
    assert rmse["C"] < rmse["B"] < rmse["A"]


def test_model_c_transfers_without_degrading(day_a_comparison, day_b_comparison):
    """C's day-B RMSE stays within 1.5x its day-A RMSE (published: 1.1x)."""
    ratio = day_b_comparison.mean("C", "rmse_K") / day_a_comparison.mean("C", "rmse_K")
    assert ratio <= 1.5, f"Model C degraded by {ratio:.2f}x on the unseen day"
