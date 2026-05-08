import sys
sys.argv = [
    "gtauav-train-baseline",
    "--mode",
    "single",
    "--split",
    "train",
    "--dataset-root",
    "University-Release",
    "--homography-precompute",
    "data/homography_university_train",
    "--homography-loss-weight",
    "1.0",
    "--out-dir",
    "runs/baseline_test",
    "--warp-backend",
    "cpu",
    "--epochs",
    "1",
    "--batch-size",
    "1",
    "--num-workers",
    "0",
    "--seed",
    "42",
]

from gtauav_loc import train_baseline
train_baseline.main()
