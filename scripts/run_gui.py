"""
Launch the Office Items Classifier GUI

Usage:
    python scripts/run_gui.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.gui_app import main

if __name__ == "__main__":
    main()