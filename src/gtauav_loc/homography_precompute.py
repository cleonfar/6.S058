"""Precompute per-sample homography targets for localization training.

This script estimates a homography for each sample in a split and writes a
single compressed cache file that training code can consume later.

Example:
    python -m gtauav_loc.homography_precompute --split train --dataset-root University-Release --out-dir data/homography_university_train
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .data import load_split
from .homography import estimate_homography_from_paths


def _process_one(idx: int, drone_p: str, sat_p: str, sift_backend: str) -> tuple[int, "np.ndarray | None", dict]:
    """Module-level worker for ProcessPoolExecutor (must be picklable on Windows)."""
    return idx, *estimate_homography_from_paths(
        Path(drone_p), Path(sat_p), sift_backend=sift_backend, return_diagnostics=True
    )


def _resolve_satellite_path(satellite_dir: Path, filename: str) -> Path:
    """Resolve the satellite image path, handling the University-1652 layout.

    University-1652 stores each satellite image in its own subdirectory:
        train/satellite/1318/1318.jpg
    but the JSON records only the bare filename ("1318.jpg") under a shared
    sate_img_dir ("train/satellite").  Try the direct path first and fall back
    to <satellite_dir>/<stem>/<filename> when the direct path does not exist.
    """
    direct = satellite_dir / filename
    if direct.exists():
        return direct
    stem = Path(filename).stem
    nested = satellite_dir / stem / filename
    if nested.exists():
        return nested
    # Return the direct path anyway so downstream code can report it as missing.
    return direct


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute homography targets for a dataset split")
    parser.add_argument("--split", default="train", help="Split JSON or split name (train/test)")
    parser.add_argument("--dataset-root", default="dataset", help="Dataset root directory")
    parser.add_argument("--out-dir", default="data/homography_precompute", help="Output directory")
    parser.add_argument("--precompute-dir", dest="out_dir", help="Alias for --out-dir")
    parser.add_argument("--write-composites", action="store_true", help="Write classical alignment overlays of drone warped onto satellite")
    parser.add_argument("--composite-dir", default=None, help="Directory for composite overlays (defaults to <out-dir>/composites)")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Blend factor for warped drone overlay in composites")
    parser.add_argument("--sift-backend", default="opencv", choices=["auto", "kornia_sift", "opencv"], help="SIFT backend: 'opencv' (default) uses CPU SIFT, 'auto'/'kornia_sift' uses Kornia GPU SIFT (slower for single-image workloads)")
    parser.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() - 1), help="Parallel worker processes for SIFT matching (default: cpu_count-1)")
    args = parser.parse_args()

    samples = load_split(args.split, args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    composite_dir = Path(args.composite_dir) if args.composite_dir else (out_dir / "composites")
    if args.write_composites:
        composite_dir.mkdir(parents=True, exist_ok=True)

    homographies = np.zeros((len(samples), 3, 3), dtype=np.float32)
    valid = np.zeros((len(samples),), dtype=bool)
    inliers_arr = np.zeros((len(samples),), dtype=np.int32)
    good_matches_arr = np.zeros((len(samples),), dtype=np.int32)
    drone_paths: list[str] = []
    sat_paths: list[str] = []

    start = time.perf_counter()

    # Build the work list first so indices are stable.
    work: list[tuple[int, Path, Path]] = []
    for index, sample in enumerate(samples):
        positive_names = list(sample.pair_pos_sate_img_list)
        drone_paths.append(str(sample.drone_img_path))
        if not positive_names:
            sat_paths.append("")
            continue
        sat_path = _resolve_satellite_path(sample.satellite_dir, positive_names[0])
        sat_paths.append(str(sat_path))
        work.append((index, sample.drone_img_path, sat_path))

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(_process_one, idx, str(drone_p), str(sat_p), args.sift_backend): idx
            for idx, drone_p, sat_p in work
        }
        progress = tqdm(as_completed(futures), total=len(futures), desc="Homographies", unit="sample")
        for future in progress:
            index, H, info = future.result()
            good_matches_arr[index] = int(info.get("good_matches", 0))
            inliers_arr[index] = int(info.get("inliers", 0))
            if H is not None:
                homographies[index] = H.astype(np.float32)
                valid[index] = True
                if args.write_composites:
                    write_overlay_composite(
                        Path(drone_paths[index]), Path(sat_paths[index]),
                        H, composite_dir, index, args.overlay_alpha,
                    )

    cache_path = out_dir / "homographies.npz"
    np.savez_compressed(
        cache_path,
        homographies=homographies,
        valid=valid,
        inliers=inliers_arr,
        good_matches=good_matches_arr,
        drone_paths=np.array(drone_paths, dtype=object),
        sat_paths=np.array(sat_paths, dtype=object),
    )

    elapsed = time.perf_counter() - start
    print(f"Wrote {cache_path} with {int(valid.sum())}/{len(samples)} valid homographies in {elapsed:.1f}s")
    if args.write_composites:
        print(f"Saved overlay composites to {composite_dir}")


def write_overlay_composite(
    drone_path: Path,
    sat_path: Path,
    homography: np.ndarray,
    composite_dir: Path,
    index: int,
    alpha: float,
) -> None:
    drone = cv2.imread(str(drone_path), cv2.IMREAD_COLOR)
    sat = cv2.imread(str(sat_path), cv2.IMREAD_COLOR)
    if drone is None or sat is None:
        return

    sat_h, sat_w = sat.shape[:2]
    warped_drone = cv2.warpPerspective(drone, homography, (sat_w, sat_h))
    valid_mask = cv2.warpPerspective(
        np.full((drone.shape[0], drone.shape[1]), 255, dtype=np.uint8),
        homography,
        (sat_w, sat_h),
    )
    composite = sat.copy()
    blend = cv2.addWeighted(sat, 1.0 - alpha, warped_drone, alpha, 0.0)
    composite[valid_mask > 0] = blend[valid_mask > 0]

    stem = f"{index:06d}_{drone_path.stem}_on_{sat_path.stem}.jpg"
    cv2.imwrite(str(composite_dir / stem), composite)


if __name__ == "__main__":
    main()