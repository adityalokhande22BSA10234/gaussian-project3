#!/usr/bin/env python3
"""CLI entry point: train the beam regression CNN.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --epochs 20 --backbone resnet18
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train import main

if __name__ == "__main__":
    main()
