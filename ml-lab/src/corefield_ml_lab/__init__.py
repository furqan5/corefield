"""Quarantined transformer hot-spot falsification harness.

Nothing in this package is validated for operational loading decisions.
"""

from __future__ import annotations

# Apply CPU/one-thread process settings before NumPy, SciPy, or Torch can be
# imported by any package submodule.  Primary CLI runs therefore inherit the
# same deterministic, memory-conservative numerical runtime.
from .runtime import enforce_cpu_only_environment as _enforce_cpu_only_environment

_enforce_cpu_only_environment()
del _enforce_cpu_only_environment

__all__ = ["__version__"]

__version__ = "0.1.0.dev0"
