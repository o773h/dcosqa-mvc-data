#!/usr/bin/env python3
"""Regenerate Supplementary Figures S12--S13 from verified simulation results.

The full coherent-state calculation is intentionally not repeated by the
top-level figure runner because it is substantially slower than replotting.
Run ``01_run_idealized_closed_system.py all --sizes 12 14 16 18 20`` followed
by ``01_run_idealized_closed_system.py extended --sizes 14 16 18 20`` to recompute
the state evolution and every summary from the fixed seed pools.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
DERIVED = ANALYSIS / "derived"
CURATED = ANALYSIS / "curated"
SCRIPT = HERE / "01_run_idealized_closed_system.py"


def main() -> int:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "plot", "--outdir", str(DERIVED)],
        check=False,
    )
    if completed.returncode:
        return completed.returncode

    CURATED.mkdir(parents=True, exist_ok=True)
    pairs = {
        "fig_supplement_simulation": "idealized_closed_system_endpoint",
        "fig_mechanism_sensitivity": "idealized_closed_system_sensitivity",
    }
    for source_stem, target_stem in pairs.items():
        for suffix in (".pdf", ".png"):
            source = DERIVED / f"{source_stem}{suffix}"
            target = CURATED / f"{target_stem}{suffix}"
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(source)
            shutil.copy2(source, target)

    # The verified summaries and publication figures are the release products;
    # remove plotting-only duplicates from the derived-data directory.
    for path in DERIVED.glob("fig_*"):
        path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
