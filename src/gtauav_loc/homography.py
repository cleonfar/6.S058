from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class HomographyHead(nn.Module):
    """Predict the four destination corners of an image homography.

    Outputs 8 values representing the (x, y) destination of each source corner
    in normalised [0, 1] image space.  Source corners are always at the image
    boundary: (0,0), (1,0), (1,1), (0,1).

    Using corner-space predictions instead of raw H-matrix entries removes the
    need to balance per-entry scales (translations vs. perspective coefficients
    differ by ~3 orders of magnitude) and keeps values well within float16 range
    so no inf losses occur during AMP training.

    tanh * _SCALE centres the output on the identity corners and allows each
    corner to move up to _SCALE away, covering large perspective transforms
    (e.g. drone-to-satellite) without exploding gradients.
    """

    # Identity corners flattened: (0,0),(1,0),(1,1),(0,1)
    _IDENTITY = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    # Maximum per-axis displacement from identity in normalised coords.
    # 1.5 allows corners to map to [-0.5, 2.5], covering the typical
    # drone-to-satellite transform range.
    _SCALE = 1.5

    def __init__(self, in_features: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 8),
        )
        # Zero weights + zero bias so tanh(0)*scale + identity = identity at init.
        nn.init.xavier_uniform_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # B x F -> B x 8 (normalised corner destinations)
        raw = self.net(features.view(features.shape[0], -1))
        identity = torch.tensor(self._IDENTITY, device=features.device, dtype=features.dtype)
        return identity + torch.tanh(raw) * self._SCALE


class SpatialHomographyHead(nn.Module):
    """Predict 4 destination corners from a spatial feature volume.

    Input  : (B, in_channels, H, W) — typically the concat of drone+satellite
             cross-attended feature maps, e.g. (B, 512, 14, 14).
    Output : (B, 8) corner destinations in normalised [0, 1] space, identity-
             centred via tanh * _SCALE.

    Uses a small ConvNet (BN + ReLU) to retain spatial structure while keeping
    the final regressor input small (32-dim after global pool).  Dropout in the
    final stage acts as the main regularizer for the regression head.
    """

    _IDENTITY = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    _SCALE = 1.5

    def __init__(self, in_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B, 32, 1, 1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32, 8),
        )
        nn.init.xavier_uniform_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        # (B, in_channels, H, W) -> (B, 8)
        x = self.conv(volume)
        x = self.pool(x)
        raw = self.regressor(x)
        identity = torch.tensor(self._IDENTITY, device=volume.device, dtype=volume.dtype)
        return identity + torch.tanh(raw) * self._SCALE


def params_to_homography(params: torch.Tensor) -> torch.Tensor:
    """Convert Bx8 params tensor to Bx3x3 homography matrices (torch.Tensor).

    params order: [h00,h01,h02,h10,h11,h12,h20,h21]
    bottom-right element is 1.
    """
    b = params.shape[0]
    device = params.device
    H = torch.zeros((b, 3, 3), device=device, dtype=params.dtype)
    # Parameterization is identity-centered: the network predicts deltas from I.
    H[:, 0, 0] = 1.0 + params[:, 0]
    H[:, 0, 1] = params[:, 1]
    H[:, 0, 2] = params[:, 2]
    H[:, 1, 0] = params[:, 3]
    H[:, 1, 1] = 1.0 + params[:, 4]
    H[:, 1, 2] = params[:, 5]
    H[:, 2, 0] = params[:, 6]
    H[:, 2, 1] = params[:, 7]
    H[:, 2, 2] = 1.0
    return H


def apply_homography_to_points(H: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply homography H (B x 3 x 3) to points (B x N x 2), returns B x N x 2.
    """
    b, n, _ = points.shape
    device = H.device
    homo = torch.cat([points, torch.ones((b, n, 1), device=device, dtype=points.dtype)], dim=-1)  # B x N x 3
    res = torch.bmm(homo, H.transpose(1, 2))  # B x N x 3
    denom = res[..., 2:3]
    denom = torch.where(denom.abs() < 1e-6, denom.sign().clamp(min=-1.0, max=1.0) * 1e-6 + (denom == 0).to(denom.dtype) * 1e-6, denom)
    res = res[..., :2] / denom
    return res


def compute_homography_reprojection_loss(
    pred_corners: torch.Tensor,
    target_H: torch.Tensor,
    image_size: Tuple[int, int],
    confidence_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reprojection loss between predicted corner destinations and target homography.

    pred_corners       : B x 8      — predicted (x, y) corner destinations in
                                      normalised [0, 1] space, ordered
                                      (0,0),(1,0),(1,1),(0,1).
    target_H           : B x 3 x 3  — ground-truth homography matrices.
    image_size         : (width, height) in pixels.
    confidence_weights : optional B-length tensor of per-sample weights.
                         Use 1.0 for SIFT-validated targets and a smaller value
                         (e.g. 0.1) for warp-based fallback targets.
                         If None, all samples are weighted equally.

    Returns a scalar loss in normalised-coordinate units (roughly 0–1 range).
    """
    b = pred_corners.shape[0]
    device = pred_corners.device
    dtype = pred_corners.dtype
    w, h = float(image_size[0]), float(image_size[1])

    # Source corners in pixel space: (0,0),(w,0),(w,h),(0,h)
    src_px = torch.tensor(
        [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], device=device, dtype=dtype
    ).unsqueeze(0).expand(b, -1, -1)

    # Apply target H to get target corners in pixel space, then normalise.
    tgt_px = apply_homography_to_points(target_H.to(device=device, dtype=dtype), src_px)
    scale = torch.tensor([[w, h]], device=device, dtype=dtype)
    tgt_norm = tgt_px / scale  # B x 4 x 2
    # Clamp to a float16-safe range (well below 65504) before computing loss.
    tgt_norm = torch.nan_to_num(tgt_norm, nan=2.0, posinf=2.0, neginf=-1.0)

    pred_norm = pred_corners.view(b, 4, 2)
    diff = pred_norm - tgt_norm
    per_sample = torch.norm(diff, dim=-1).mean(dim=-1)  # B

    if confidence_weights is not None:
        w = confidence_weights.to(device=device, dtype=dtype)
        loss = (per_sample * w).sum() / w.sum().clamp(min=1e-6)
    else:
        loss = per_sample.mean()
    return loss


def _prepare_gray_for_matching(image_bgr: np.ndarray) -> np.ndarray:
    """Convert to grayscale and equalize local contrast.

    Drone and satellite imagery typically differ strongly in exposure and white
    balance. CLAHE produces an illumination-robust signal so SIFT can find the
    same stable structures (building corners, road junctions, etc.) in both.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _make_sift() -> "cv2.Feature2D":
    # Slightly relaxed thresholds + a generous feature budget. The extra
    # candidates feed Lowe's ratio test, which then filters weak matches.
    kwargs = dict(nfeatures=4000, contrastThreshold=0.03, edgeThreshold=12, sigma=1.6)
    try:
        return cv2.SIFT_create(**kwargs)
    except Exception:
        return cv2.xfeatures2d.SIFT_create(**kwargs)


def _is_homography_well_conditioned(
    H: np.ndarray,
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
    max_scale_ratio: float = 8.0,
    corner_margin_factor: float = 0.5,
) -> bool:
    """Reject degenerate homographies before they pollute downstream caches.

    Checks: finiteness, non-singularity, no reflection (positive det of upper
    2x2), bounded affine scale change, and that the four warped drone corners
    land inside (or near) the satellite image extent. Without these checks
    near-collinear inlier sets produce H matrices with norms in the millions
    that warp the drone image to a single pixel or off the canvas entirely.
    """
    if H is None or not np.all(np.isfinite(H)):
        return False
    # Bottom-right normalization for stable inspection.
    if abs(H[2, 2]) < 1e-8:
        return False
    Hn = H / H[2, 2]
    A = Hn[:2, :2]
    det_a = float(np.linalg.det(A))
    if det_a <= 0:  # reflection / degenerate
        return False
    # Singular values bound the per-axis scale change.
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[-1] < 1e-6:
        return False
    if sv[0] / sv[-1] > max_scale_ratio:
        return False

    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    corners = np.float32([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, Hn).reshape(-1, 2)
    if not np.all(np.isfinite(warped)):
        return False
    # Warped quad must have positive area and stay within a generous margin
    # around the satellite tile (allow some overshoot for off-center crops).
    margin_w, margin_h = corner_margin_factor * dst_w, corner_margin_factor * dst_h
    if warped[:, 0].min() < -margin_w or warped[:, 0].max() > dst_w + margin_w:
        return False
    if warped[:, 1].min() < -margin_h or warped[:, 1].max() > dst_h + margin_h:
        return False
    # Quad area via the shoelace formula — guards against collapsed shapes.
    x = warped[:, 0]
    y = warped[:, 1]
    area = 0.5 * abs(x[0] * (y[1] - y[3]) + x[1] * (y[2] - y[0]) + x[2] * (y[3] - y[1]) + x[3] * (y[0] - y[2]))
    if area < 0.01 * src_w * src_h:
        return False
    return True


def estimate_homography_from_paths(
    drone_path: Path,
    sat_path: Path,
    ratio: float = 0.75,
    ransac_thresh: float = 4.0,
    min_inliers: int = 15,
    sift_backend: str = "auto",
    return_diagnostics: bool = False,
    max_scale_ratio: float = 8.0,
    corner_margin_factor: float = 0.5,
):
    """Estimate homography H (3x3 numpy) using SIFT + RANSAC.

    By default uses ``sift_backend="auto"`` which resolves to Kornia's
    GPU-accelerated SIFT (``kornia_sift``) so the GPU is utilised during
    the precompute phase.  Pass ``sift_backend="opencv"`` to force the CPU
    path (useful on machines without a CUDA-capable GPU).

    Pipeline: SIFT detect & describe -> BFMatcher with Lowe's ratio test ->
    RANSAC homography -> geometric sanity check. Returns the homography or
    ``None`` if any stage fails.

    With ``return_diagnostics=True`` returns a ``(H, info)`` tuple where
    ``info`` includes match/inlier counts useful for debugging and ranking.
    """
    drone = cv2.imread(str(drone_path), cv2.IMREAD_COLOR)
    sat = cv2.imread(str(sat_path), cv2.IMREAD_COLOR)

    info = {"good_matches": 0, "inliers": 0, "reason": "", "backend": sift_backend}

    def _fail(reason: str):
        info["reason"] = reason
        return (None, info) if return_diagnostics else None

    if drone is None or sat is None:
        return _fail("imread_failed")

    if sift_backend == "opencv":
        # CPU path: CLAHE normalisation + OpenCV SIFT.
        gray1 = _prepare_gray_for_matching(drone)
        gray2 = _prepare_gray_for_matching(sat)
        sift = _make_sift()
        kps1_cv, des1 = sift.detectAndCompute(gray1, None)
        kps2_cv, des2 = sift.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or len(kps1_cv) < min_inliers or len(kps2_cv) < min_inliers:
            return _fail("insufficient_keypoints")
        kp1 = np.float32([kp.pt for kp in kps1_cv])
        kp2 = np.float32([kp.pt for kp in kps2_cv])
    else:
        # GPU path: Kornia SIFT (falls back to CPU if CUDA unavailable).
        from .local_features import extract_sift_features
        feat1 = extract_sift_features(drone, backend=sift_backend)
        feat2 = extract_sift_features(sat, backend=sift_backend)
        des1 = feat1.descriptors
        des2 = feat2.descriptors
        info["backend"] = feat1.backend
        if des1 is None or des2 is None or len(feat1.keypoints) < min_inliers or len(feat2.keypoints) < min_inliers:
            return _fail("insufficient_keypoints")
        kp1 = feat1.keypoints[:, :2]  # x, y
        kp2 = feat2.keypoints[:, :2]

    # NORM_L2 is correct for SIFT descriptors. crossCheck and knnMatch are
    # mutually exclusive — Lowe's ratio test is the filter here.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    info["good_matches"] = len(good)
    if len(good) < min_inliers:
        return _fail("insufficient_good_matches")

    pts1 = np.float32([kp1[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(
        pts1,
        pts2,
        cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=5000,
        confidence=0.999,
    )
    if H is None or mask is None:
        return _fail("ransac_failed")

    inliers = int(mask.sum())
    info["inliers"] = inliers
    if inliers < min_inliers:
        return _fail("too_few_inliers")

    if not _is_homography_well_conditioned(
        H,
        src_size=(drone.shape[1], drone.shape[0]),
        dst_size=(sat.shape[1], sat.shape[0]),
        max_scale_ratio=max_scale_ratio,
        corner_margin_factor=corner_margin_factor,
    ):
        return _fail("degenerate_homography")

    H = H.astype(np.float32)
    info["reason"] = "ok"
    return (H, info) if return_diagnostics else H
