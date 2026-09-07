#!/usr/bin/env python3
"""CLI entry point: generate the synthetic dataset.

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --normal 200 --tilted 25 --noisy 50 --background 25
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic.dataset_builder import main

if __name__ == "__main__":
    main()
