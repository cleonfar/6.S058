"""Align a single drone image to a satellite tile via classical SIFT+RANSAC.

Usage:
    python scripts/align_drone_to_satellite.py <drone.png> <sat.png> [--out composite.jpg]

This is the minimal end-to-end pipeline described in the problem statement:
detect SIFT keypoints in both images, match them with Lowe's ratio test,
robustly fit a 3x3 homography with RANSAC, then warp the drone image into
the satellite frame and blend them into a composite overlay.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gtauav_loc.homography import estimate_homography_from_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drone", type=Path)
    parser.add_argument("satellite", type=Path)
    parser.add_argument("--out", type=Path, default=Path("composite.jpg"))
    parser.add_argument("--alpha", type=float, default=0.45, help="Drone overlay opacity")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=4.0)
    args = parser.parse_args()

    H, info = estimate_homography_from_paths(
        args.drone,
        args.satellite,
        ratio=args.ratio,
        ransac_thresh=args.ransac_threshold,
        return_diagnostics=True,
    )
    print(f"good_matches={info['good_matches']} inliers={info['inliers']} status={info['reason']}")
    if H is None:
        raise SystemExit("Alignment failed; no composite written.")

    drone = cv2.imread(str(args.drone), cv2.IMREAD_COLOR)
    sat = cv2.imread(str(args.satellite), cv2.IMREAD_COLOR)
    sat_h, sat_w = sat.shape[:2]
    warped = cv2.warpPerspective(drone, H, (sat_w, sat_h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.full(drone.shape[:2], 255, dtype=np.uint8), H, (sat_w, sat_h)
    )
    composite = sat.copy()
    blended = cv2.addWeighted(sat, 1.0 - args.alpha, warped, args.alpha, 0.0)
    composite[mask > 0] = blended[mask > 0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), composite)
    print(f"Wrote {args.out}")
    print("H =\n" + np.array2string(H, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
