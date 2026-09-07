#!/usr/bin/env python3
"""CLI entry point: save a preview grid of sample dataset images.

Usage:
    python scripts/preview_dataset.py --output preview.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.preview import main

if __name__ == "__main__":
    main()
