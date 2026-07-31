from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so `src` module is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
