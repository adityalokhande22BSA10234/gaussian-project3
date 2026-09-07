#!/usr/bin/env python3
"""CLI entry point: run inference on a real beam-camera image.

Usage:
    python scripts/run_inference.py --image path/to/real_capture.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.infer import main

if __name__ == "__main__":
    main()
