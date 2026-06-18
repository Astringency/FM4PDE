"""Deprecated compatibility wrapper for :mod:`sampling.aggregate`."""

from __future__ import annotations

from sampling import aggregate as _aggregate

globals().update({name: value for name, value in vars(_aggregate).items() if not name.startswith("__")})


if __name__ == "__main__":
    _aggregate.main()
