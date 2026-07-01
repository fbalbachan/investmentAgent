"""Ensure the project root is importable so tests can `import agent` / `import main`."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
