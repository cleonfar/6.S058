"""Local feature extraction backends for SIFT-style descriptors.

The project prefers a CUDA-backed implementation when available, but falls back
to OpenCV's CPU SIFT so the pipeline remains runnable on systems where
pypopsift cannot be built.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import cv2
import numpy as np


SiftBackend = Literal["auto", "opencv", "pypopsift", "superpoint", "kornia_sift"]


@dataclass(frozen=True)
class SiftFeatures:
    keypoints: np.ndarray
    descriptors: np.ndarray | None
    backend: str


def extract_sift_features(
    image_bgr: np.ndarray,
    backend: SiftBackend = "auto",
    peak_threshold: float = 0.1,
    edge_threshold: float = 10.0,
    target_num_features: int = 8000,
    feature_process_size: int = 2048,
) -> SiftFeatures:
    if backend not in ("auto", "opencv", "pypopsift", "superpoint", "kornia_sift"):
        raise ValueError(f"Unsupported SIFT backend: {backend}")

    # `auto` is now committed to `kornia_sift` — do not silently fall back.
    if backend == "auto":
        backend = "kornia_sift"

    if backend == "pypopsift":
        return _extract_with_pypopsift(
            image_bgr,
            peak_threshold=peak_threshold,
            edge_threshold=edge_threshold,
            target_num_features=target_num_features,
            feature_process_size=feature_process_size,
        )

    if backend == "kornia_sift":
        return _extract_with_kornia_sift(
            image_bgr, target_num_features=target_num_features, feature_process_size=feature_process_size
        )

    if backend == "superpoint":
        return _extract_with_superpoint(
            image_bgr, target_num_features=target_num_features, feature_process_size=feature_process_size
        )

    if backend == "opencv":
        return _extract_with_opencv(image_bgr)

    # Should not reach here due to earlier validation.
    raise RuntimeError(f"Unhandled SIFT backend: {backend}")


def _extract_with_opencv(image_bgr: np.ndarray) -> SiftFeatures:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV was installed without SIFT support. Install opencv-contrib-python instead of opencv-python.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    keypoint_array = (
        np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints], dtype=np.float32)
        if keypoints
        else np.zeros((0, 4), dtype=np.float32)
    )
    return SiftFeatures(keypoints=keypoint_array, descriptors=descriptors, backend="opencv")


def _extract_with_pypopsift(
    image_bgr: np.ndarray,
    peak_threshold: float,
    edge_threshold: float,
    target_num_features: int,
    feature_process_size: int,
) -> SiftFeatures:
    from pypopsift import popsift

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 image array, got shape {image_bgr.shape}")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = _resize_for_processing(rgb, feature_process_size)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    points, descriptors = popsift(
        gray.astype(np.uint8),
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
        target_num_features=target_num_features,
    )

    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        points = np.zeros((0, 4), dtype=np.float32)
    elif points.ndim == 1:
        points = points.reshape(1, -1)

    descriptors = None if descriptors is None else np.asarray(descriptors, dtype=np.float32)
    return SiftFeatures(keypoints=points, descriptors=descriptors, backend="pypopsift")


def _resize_for_processing(image_rgb: np.ndarray, max_size: int) -> np.ndarray:
    if max_size <= 0:
        return image_rgb

    height, width = image_rgb.shape[:2]
    size = max(height, width)
    if size <= max_size:
        return image_rgb

    scale = max_size / size
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _extract_with_superpoint(
    image_bgr: np.ndarray, *, target_num_features: int, feature_process_size: int
) -> SiftFeatures:
    """Extract features using a SuperPoint-style learned backend.

    This backend requires `torch` and a library that exposes a SuperPoint
    implementation such as `kornia` (recommended). If the required packages
    are not installed a helpful RuntimeError is raised describing how to add
    them.
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("DISK backend requires PyTorch: pip install torch torchvision") from e

    try:
        from kornia.feature.disk import DISK
    except Exception:
        raise RuntimeError(
            "DISK backend requires the 'kornia' package with DISK support.\nInstall with: pip install kornia"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare RGB tensor [B,3,H,W] normalized to [0,1]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = _resize_for_processing(rgb, feature_process_size)
    img_t = torch.from_numpy(rgb.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # Load pretrained DISK (epipolar model is general-purpose)
    try:
        model = DISK.from_pretrained("epipolar", device=device)
    except Exception:
        model = DISK().to(device).eval()

    with torch.no_grad():
        features = model(img_t, n=int(target_num_features))

    if not features:
        return SiftFeatures(keypoints=np.zeros((0, 4), dtype=np.float32), descriptors=np.zeros((0, 0), dtype=np.float32), backend="disk")

    f = features[0]
    kp = f.keypoints.cpu().numpy()
    desc = f.descriptors.cpu().numpy()

    # Convert keypoints to Nx4 (x, y, size, angle)
    if kp.size == 0:
        kp4 = np.zeros((0, 4), dtype=np.float32)
    else:
        kp4 = np.zeros((kp.shape[0], 4), dtype=np.float32)
        kp4[:, 0:2] = kp
        kp4[:, 2] = 1.0
        kp4[:, 3] = 0.0

    return SiftFeatures(keypoints=kp4.astype(np.float32), descriptors=desc.astype(np.float32), backend="disk")


def _extract_with_kornia_sift(
    image_bgr: np.ndarray, *, target_num_features: int, feature_process_size: int
) -> SiftFeatures:
    """Extract features using Kornia's GPU-accelerated SIFT implementation.

    This backend requires PyTorch and Kornia with SIFT support.
    It provides GPU acceleration for both detection and descriptor computation.
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("Kornia SIFT backend requires PyTorch: pip install torch torchvision") from e

    try:
        from kornia.feature import ScaleSpaceDetector, SIFTDescriptor, get_laf_descriptors, get_laf_center, get_laf_scale, get_laf_orientation
    except Exception:
        raise RuntimeError(
            "Kornia SIFT backend requires the 'kornia' package with SIFT support.\nInstall with: pip install kornia"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare grayscale tensor [B,1,H,W] normalized to [0,1]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = _resize_for_processing(rgb, feature_process_size)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_t = torch.from_numpy(gray.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    # Create SIFT detector
    detector = ScaleSpaceDetector(num_features=target_num_features, mr_size=6.0).to(device)

    with torch.inference_mode():
        lafs, responses = detector(gray_t)

    if lafs.numel() == 0:
        return SiftFeatures(keypoints=np.zeros((0, 4), dtype=np.float32), descriptors=np.zeros((0, 128), dtype=np.float32), backend="kornia_sift")

    # Compute SIFT descriptors using get_laf_descriptors
    sift_descriptor = SIFTDescriptor(patch_size=32).to(device)
    descriptors = get_laf_descriptors(gray_t, lafs, sift_descriptor, patch_size=32, grayscale_descriptor=True)
    
    if descriptors.numel() == 0:
        descriptors_np = np.zeros((lafs.shape[1], 128), dtype=np.float32)
    else:
        descriptors_np = descriptors[0].detach().cpu().numpy().astype(np.float32)

    # Convert LAF to keypoint format (x, y, size, angle)
    centers = get_laf_center(lafs)  # Shape: (B, N, 2)
    scales = get_laf_scale(lafs)     # Shape: (B, N, 1, 1)
    angles = get_laf_orientation(lafs)  # Shape: (B, N, 1)
    
    centers_np = centers[0].cpu().numpy().astype(np.float32)  # (N, 2)
    scales_np = scales[0, :, 0, 0].cpu().numpy().astype(np.float32)  # (N,)
    angles_np = angles[0, :, 0].cpu().numpy().astype(np.float32)  # (N,)
    
    num_kps = centers_np.shape[0]
    kp4 = np.zeros((num_kps, 4), dtype=np.float32)
    kp4[:, 0:2] = centers_np  # x, y
    kp4[:, 2] = scales_np  # size
    kp4[:, 3] = angles_np  # angle

    return SiftFeatures(keypoints=kp4, descriptors=descriptors_np, backend="kornia_sift")
