#!/usr/bin/env python
"""Quick 2-3 epoch test run without overhead warping to diagnose embedding quality."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gtauav_loc.train_baseline import main as train_main

if __name__ == "__main__":
    # Override sys.argv to run a short test without warp
    sys.argv = [
        "train_no_warp",
        "--split", "dataset/same-area-drone2sate-train.json",
        "--dataset-root", "dataset",
        "--precompute-dir", "data/precompute",
        "--out-dir", "runs/baseline-no-warp-test",
        "--use-gpu",
        "--warp-backend", "torch",
        "--skip-warp",
        "--epochs", "3",
        "--batch-size", "32",
    ]
    train_main()
