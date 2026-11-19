#!/usr/bin/env python
"""Ciphertext-only attack on block-wise EtC images; run ``python run_attack.py --help``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run from a fresh clone

from ejps.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
