"""Make the C++ pybind modules importable regardless of the current directory."""

import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
_BUILD = PROJ_ROOT / "build"

for subdir in ("engine", "training"):
    path = str(_BUILD / subdir)
    if path not in sys.path:
        sys.path.insert(0, path)
