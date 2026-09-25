"""Analyses behind the manuscript numbers; see analysis/README.md.

Inputs that are not in the public release are read from OVARIAN_PRIVATE (default: private/ in the
repository); outputs are written to OVARIAN_RESULTS/analysis.
"""
import json
import os
from pathlib import Path

from ovarian import paths


def private_dir() -> Path:
    return Path(os.environ.get("OVARIAN_PRIVATE", paths.REPO / "private"))


def private_input(*parts: str) -> Path:
    """Path to a non-public input; raises FileNotFoundError if it is absent."""
    path = private_dir().joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"OVARIAN_PRIVATE/{'/'.join(parts)} is missing. This input is not part of the "
                                "public release; it is available from the corresponding author on reasonable request.")
    return path


def output_dir(*parts: str) -> Path:
    return paths.results_dir("analysis", *parts)


def write_json(obj, name: str) -> Path:
    path = output_dir() / name
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    return path
