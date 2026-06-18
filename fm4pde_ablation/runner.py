"""Compatibility wrapper for :mod:`sampling.runner`."""

from __future__ import annotations

from sampling import runner as _runner

globals().update({name: value for name, value in vars(_runner).items() if not name.startswith("__")})


if __name__ == "__main__":
    _runner.main()
