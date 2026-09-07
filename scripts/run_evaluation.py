#!/usr/bin/env python3
"""CLI entry point: evaluate the trained model on a dataset split.

Usage:
    python scripts/run_evaluation.py --split test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.evaluate import main

if __name__ == "__main__":
    main()
