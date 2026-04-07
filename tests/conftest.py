"""Pytest configuration — adds src to path."""

import sys
from pathlib import Path

# Add src to path so hprobes is importable without installation
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
