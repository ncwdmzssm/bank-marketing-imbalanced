"""Entry point to run effect experiments from the project root."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Allow importing the effect package when invoked as a script."""
    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()
    from effect.run_effect import main as run_main

    run_main()


if __name__ == "__main__":
    main()

