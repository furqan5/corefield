from __future__ import annotations

import pytest

from corefield_ml_lab.__main__ import _confirmation_trigger, _parser
from corefield_ml_lab.confirmation import ConfirmationTrigger


def test_cli_exposes_preflight_and_primary_order_commands() -> None:
    parser = _parser()
    assert parser.parse_args(["preflight"]).command == "preflight"
    assert parser.parse_args(["e1"]).command == "e1"
    assert parser.parse_args(["e3"]).command == "e3"
    assert parser.parse_args(["e2"]).command == "e2"
    assert parser.parse_args(["e4"]).command == "e4"
    assert parser.parse_args(["e5"]).command == "e5"
    args = parser.parse_args(
        [
            "confirmation",
            "e3",
            "--method",
            "pinn",
            "--target-load",
            "1.30",
            "--prior-run-id",
            "e3-example",
        ]
    )
    assert _confirmation_trigger(args) == ConfirmationTrigger.e3("pinn", 1.30)


def test_cli_requires_known_command() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["unknown"])
