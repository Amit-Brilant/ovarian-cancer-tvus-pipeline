"""Write checkpoints atomically on the destination filesystem."""

import os
import shutil
import tempfile
from pathlib import Path


def save_checkpoint_atomic(state, destination, save):
    """Keep the previous best weights intact if serialization or disk writes fail."""
    destination = Path(destination)
    if destination.is_symlink():
        raise RuntimeError(f"Refusing a symlink checkpoint destination: {destination}")
    required = sum(value.numel() * value.element_size() for value in state.values())
    reserve = 128 * 1024 * 1024
    if shutil.disk_usage(destination.parent).free < required + reserve:
        raise OSError("Insufficient persistent storage for an atomic checkpoint save")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".", suffix=".partial", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
