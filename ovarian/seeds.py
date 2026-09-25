"""Random seed handling. No seed value is stored in the code: it is given at run time,
with --seed on the command line or the SEED environment variable."""
import os

_current: int | None = None


def resolve(seed: int | None = None) -> int:
    """Return the run seed: the argument if given, else the one already set, else $SEED."""
    global _current
    if seed is None:
        seed = _current if _current is not None else os.environ.get("SEED")
    if seed is None:
        raise SystemExit("no random seed given: pass --seed or set the SEED environment variable")
    _current = int(seed)
    return _current
