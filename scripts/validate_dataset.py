#!/usr/bin/env python3
"""CLI entry point: validate the synthetic dataset (image integrity + labels).

Usage:
    python scripts/validate_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.validate import main

if __name__ == "__main__":
    main()
