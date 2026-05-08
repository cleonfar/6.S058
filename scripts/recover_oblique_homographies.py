"""Re-run SIFT+RANSAC on samples that had enough inliers but were rejected by
the strict conditioning gate, using relaxed geometric constraints.

These are mostly oblique-angle shots where the perspective warp produces a
scale ratio or corner overshoot that exceeds the default thresholds.  Relaxing
those thresholds recovers valid homographies for roughly 3–9 k extra samples.

The script writes an updated ``homographies.npz`` (or a new path via
``--output``) with two additions:
  - recovered H matrices filled in for newly-valid samples
  - a ``confidence`` float32 array: 1.0 for strict-valid, ``--confidence``
    for relaxed-recovered, 0.0 for no-match

Example
-------
    python scripts/recover_oblique_homographies.py \\
        --npz data/homography_university_train/homographies.npz \\
        --dataset-root University-Release \\
        --min-inliers 15 \\
        --max-scale-ratio 20.0 \\
        --corner-margin-factor 1.5 \\
        --confidence 0.5 \\
        --sift-backend opencv
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gtauav_loc.homography import estimate_homography_from_paths


def _recover_one(
    idx: int,
    drone_path: str,
    sat_path: str,
    sift_backend: str,
    max_scale_ratio: float,
    corner_margin_factor: float,
):
    """Module-level worker so it can be pickled by spawn-based multiprocessing."""
    H, info = estimate_homography_from_paths(
        Path(drone_path),
        Path(sat_path),
        sift_backend=sift_backend,
        return_diagnostics=True,
        max_scale_ratio=max_scale_ratio,
        corner_margin_factor=corner_margin_factor,
    )
    return idx, H, info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover oblique homographies with relaxed conditioning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--npz", required=True,
        help="Path to existing homographies.npz produced by homography_precompute",
    )
    parser.add_argument(
        "--output", default=None,
        help="Where to write the updated npz (defaults to overwriting --npz)",
    )
    parser.add_argument(
        "--min-inliers", type=int, default=15,
        help="Only attempt recovery for samples with this many RANSAC inliers or more",
    )
    parser.add_argument(
        "--max-scale-ratio", type=float, default=20.0,
        help="Relaxed max singular-value ratio for the affine part of H (strict default is 8.0)",
    )
    parser.add_argument(
        "--corner-margin-factor", type=float, default=1.5,
        help="Relaxed corner overshoot margin as a fraction of dst size (strict default is 0.5)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Confidence weight assigned to recovered (relaxed) homographies in [0, 1]",
    )
    parser.add_argument(
        "--sift-backend", default="opencv", choices=["auto", "kornia_sift", "opencv"],
        help="SIFT backend",
    )
    parser.add_argument(
        "--num-workers", type=int, default=max(1, os.cpu_count() - 1),
        help="Parallel worker processes (default: cpu_count-1)",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    output_path = Path(args.output) if args.output else npz_path

    if not npz_path.exists():
        print(f"ERROR: npz not found: {npz_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {npz_path} …")
    data = np.load(npz_path, allow_pickle=True)
    homographies = np.asarray(data["homographies"], dtype=np.float32).copy()
    valid = np.asarray(data["valid"], dtype=bool).copy()
    inliers_arr = np.asarray(data["inliers"], dtype=np.int32).copy()
    good_matches_arr = np.asarray(data["good_matches"], dtype=np.int32).copy()
    drone_paths = np.asarray(data["drone_paths"], dtype=object)
    sat_paths = np.asarray(data["sat_paths"], dtype=object)

    # Bootstrap confidence array: 1.0 for existing strict-valid, 0.0 otherwise.
    if "confidence" in data:
        confidence = np.asarray(data["confidence"], dtype=np.float32).copy()
        print(f"  Found existing confidence array (min={confidence[valid].min():.2f} max={confidence[valid].max():.2f})")
    else:
        confidence = np.zeros(len(valid), dtype=np.float32)
        confidence[valid] = 1.0

    n_already_valid = int(valid.sum())

    # Candidates: already-invalid but had enough inliers during precompute.
    candidate_indices = np.where(~valid & (inliers_arr >= args.min_inliers))[0]
    print(
        f"  Currently valid: {n_already_valid}/{len(valid)} "
        f"  Candidates for recovery (inliers >= {args.min_inliers}): {len(candidate_indices)}"
    )

    recovered = 0
    failed_read = 0
    failed_conditioning = 0

    # Filter to candidates that have readable files.
    valid_candidates = []
    for idx in candidate_indices:
        dp = str(drone_paths[idx])
        sp = str(sat_paths[idx])
        if dp and sp and Path(dp).exists() and Path(sp).exists():
            valid_candidates.append(idx)
        else:
            failed_read += 1

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(
                _recover_one,
                idx,
                str(drone_paths[idx]),
                str(sat_paths[idx]),
                args.sift_backend,
                args.max_scale_ratio,
                args.corner_margin_factor,
            ): idx
            for idx in valid_candidates
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Recovering", unit="sample"):
            idx, H, info = future.result()
            if H is not None:
                homographies[idx] = H.astype(np.float32)
                valid[idx] = True
                confidence[idx] = float(args.confidence)
                inliers_arr[idx] = int(info.get("inliers", inliers_arr[idx]))
                recovered += 1
            else:
                failed_conditioning += 1

    print(
        f"\nRecovery complete:"
        f"\n  Newly valid:       {recovered}"
        f"\n  Failed (no file):  {failed_read}"
        f"\n  Failed (cond):     {failed_conditioning}"
        f"\n  Total valid now:   {int(valid.sum())}/{len(valid)}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        homographies=homographies,
        valid=valid,
        inliers=inliers_arr,
        good_matches=good_matches_arr,
        drone_paths=drone_paths,
        sat_paths=sat_paths,
        confidence=confidence,
    )
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
