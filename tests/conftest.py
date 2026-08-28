# Ensures the backend package is importable from the repo root when running pytest.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
