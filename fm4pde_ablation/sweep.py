"""Deprecated compatibility wrapper for :mod:`sampling.sweep`."""

from __future__ import annotations

from sampling import sweep as _sweep

globals().update({name: value for name, value in vars(_sweep).items() if not name.startswith("__")})


if __name__ == "__main__":
    _sweep.main()
