"""Approximate UAV-to-overhead warping utilities.

This is a deterministic geometry preprocessing step, not a learned model.
It uses the camera/drone pose metadata to build a rough bird's-eye view under
the flat-ground assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OverheadWarpConfig:
    output_size: Tuple[int, int] = (384, 384)
    reference_height: float = 400.0
    field_of_view_degrees: float = 60.0
    use_camera_pose: bool = True
    backend: str = "auto"
    device: str | None = None


def warp_to_overhead(
    image: Image.Image | np.ndarray,
    pose: Mapping[str, float] | None,
    config: OverheadWarpConfig | None = None,
) -> Image.Image | np.ndarray:
    """Warp a UAV image into an approximate top-down view.

    The transform is intentionally approximate:
    - pitch/roll/yaw are treated as Euler angles in degrees
    - height controls the zoom level of the output patch
    - the ground is assumed planar

    The return type matches the input type.
    """

    config = config or OverheadWarpConfig()
    pose = pose or {}

    np_image, input_was_pil = _to_numpy_rgb(image)
    height, width = np_image.shape[:2]

    roll_deg, pitch_deg, yaw_deg, altitude = _extract_pose(pose)
    if not config.use_camera_pose:
        roll_deg = pitch_deg = 0.0

    focal_length = _approximate_focal_length(width, config.field_of_view_degrees)
    intrinsic = np.array(
        [[focal_length, 0.0, width * 0.5], [0.0, focal_length, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    intrinsic_inv = np.linalg.inv(intrinsic)

    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    rotation = _rotation_matrix(roll, pitch, yaw)
    homography = intrinsic @ rotation @ intrinsic_inv

    zoom = config.reference_height / max(float(altitude), 1.0)
    scaling = np.array(
        [[zoom, 0.0, 0.0], [0.0, zoom, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    center_x = width * 0.5
    center_y = height * 0.5
    translate_to_origin = np.array(
        [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    translate_back = np.array(
        [[1.0, 0.0, config.output_size[0] * 0.5], [0.0, 1.0, config.output_size[1] * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    transform = translate_back @ scaling @ homography @ translate_to_origin

    if _should_use_torch_backend(config):
        warped = _warp_perspective_torch(np_image, transform, config.output_size, config.device)
    else:
        warped = cv2.warpPerspective(np_image, transform, config.output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    if input_was_pil:
        return Image.fromarray(warped)
    return warped


def get_warp_homography(
    pose: Mapping[str, float],
    input_size: int,
    config: OverheadWarpConfig | None = None,
) -> np.ndarray:
    """Return the 3×3 drone→overhead homography for a square input image.

    Maps pixel coordinates in a [input_size × input_size] drone image to pixel
    coordinates in the overhead view (also input_size × input_size when
    config.output_size is (input_size, input_size)).

    Used as a geometric fallback target for samples without a SIFT-validated
    homography.
    """
    config = config or OverheadWarpConfig(output_size=(input_size, input_size))
    return _build_transform_matrix(input_size, input_size, pose, config)


def batch_warp_to_overhead(
    images: torch.Tensor,
    poses: Sequence[Mapping[str, float]],
    config: OverheadWarpConfig | None = None,
) -> torch.Tensor:
    """Warp a batch of images on the current device.

    Expected input shape is BxCxHxW with values in [0, 1].
    The output uses the same dtype/device and the configured output size.
    """

    config = config or OverheadWarpConfig(backend="torch")
    if images.ndim != 4:
        raise ValueError(f"Expected a 4D BCHW tensor, got shape {tuple(images.shape)}")

    if images.shape[0] != len(poses):
        raise ValueError("Batch size and number of poses must match")

    if not torch.is_tensor(images):
        raise TypeError("images must be a torch.Tensor")

    device = images.device
    if device.type == "cpu" and config.backend != "cpu":
        # Keep the API predictable if the caller forgot to move the batch.
        device = torch.device(config.device or "cpu")
        images = images.to(device)

    output_width, output_height = config.output_size
    warped_batches = []
    for image, pose in zip(images, poses):
        transform = _build_transform_matrix(image.shape[-2], image.shape[-1], pose, config)
        warped_batches.append(_warp_torch_tensor(image.unsqueeze(0), transform, output_width, output_height))

    return torch.cat(warped_batches, dim=0)


def _extract_pose(pose: Mapping[str, float]) -> Tuple[float, float, float, float]:
    roll = _first_present(pose, ("cam_roll", "drone_roll"), default=0.0)
    pitch = _first_present(pose, ("cam_pitch", "drone_pitch"), default=0.0)
    yaw = _first_present(pose, ("cam_yaw", "drone_yaw"), default=0.0)
    height = _first_present(pose, ("height",), default=400.0)
    return roll, pitch, yaw, height


def _build_transform_matrix(height: int, width: int, pose: Mapping[str, float], config: OverheadWarpConfig) -> np.ndarray:
    roll_deg, pitch_deg, yaw_deg, altitude = _extract_pose(pose)
    if not config.use_camera_pose:
        roll_deg = pitch_deg = 0.0

    focal_length = _approximate_focal_length(width, config.field_of_view_degrees)
    intrinsic = np.array(
        [[focal_length, 0.0, width * 0.5], [0.0, focal_length, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    intrinsic_inv = np.linalg.inv(intrinsic)

    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    rotation = _rotation_matrix(roll, pitch, yaw)
    homography = intrinsic @ rotation @ intrinsic_inv

    zoom = config.reference_height / max(float(altitude), 1.0)
    scaling = np.array(
        [[zoom, 0.0, 0.0], [0.0, zoom, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    center_x = width * 0.5
    center_y = height * 0.5
    translate_to_origin = np.array(
        [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    translate_back = np.array(
        [[1.0, 0.0, config.output_size[0] * 0.5], [0.0, 1.0, config.output_size[1] * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    return translate_back @ scaling @ homography @ translate_to_origin


def _first_present(pose: Mapping[str, float], keys: Iterable[str], default: float) -> float:
    for key in keys:
        if key in pose:
            return float(pose[key])
    return float(default)


def _approximate_focal_length(image_width: int, field_of_view_degrees: float) -> float:
    fov_radians = np.deg2rad(field_of_view_degrees)
    return float((image_width * 0.5) / np.tan(fov_radians * 0.5))


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cos_r, -sin_r], [0.0, sin_r, cos_r]], dtype=np.float32)
    ry = np.array([[cos_p, 0.0, sin_p], [0.0, 1.0, 0.0], [-sin_p, 0.0, cos_p]], dtype=np.float32)
    rz = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def _warp_torch_tensor(image: torch.Tensor, transform: np.ndarray, output_width: int, output_height: int) -> torch.Tensor:
    device = image.device
    tensor = image.float().clamp(0.0, 1.0)

    ys, xs = torch.meshgrid(
        torch.linspace(0, output_height - 1, output_height, device=device),
        torch.linspace(0, output_width - 1, output_width, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    target_points = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3).T

    inverse = torch.from_numpy(np.linalg.inv(transform)).to(device=device, dtype=torch.float32)
    source_points = inverse @ target_points
    source_points = source_points[:2] / source_points[2:].clamp_min(1e-6)

    source_x = source_points[0].reshape(output_height, output_width)
    source_y = source_points[1].reshape(output_height, output_width)
    norm_x = (source_x / max(tensor.shape[-1] - 1, 1)) * 2.0 - 1.0
    norm_y = (source_y / max(tensor.shape[-2] - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

    warped = F.grid_sample(tensor, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    return warped


def _to_numpy_rgb(image: Image.Image | np.ndarray) -> Tuple[np.ndarray, bool]:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB")), True

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(image)!r}")

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected an HxWx3 image array, got shape {image.shape}")

    if image.shape[2] == 4:
        image = image[:, :, :3]

    # OpenCV expects BGR arrays; the loaded dataset images are RGB when coming from PIL.
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image, False


def _should_use_torch_backend(config: OverheadWarpConfig) -> bool:
    if config.backend == "cpu":
        return False
    if config.backend == "torch":
        return True
    return torch.cuda.is_available()


def _warp_perspective_torch(image: np.ndarray, transform: np.ndarray, output_size: Tuple[int, int], device: str | None) -> np.ndarray:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensor = torch.from_numpy(image).to(target_device, non_blocking=True).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    out_h, out_w = output_size[1], output_size[0]
    ys, xs = torch.meshgrid(
        torch.linspace(0, out_h - 1, out_h, device=target_device),
        torch.linspace(0, out_w - 1, out_w, device=target_device),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    target_points = torch.stack([xs, ys, ones], dim=-1).reshape(-1, 3).T

    inverse = torch.from_numpy(np.linalg.inv(transform)).to(target_device, dtype=torch.float32)
    source_points = inverse @ target_points
    source_points = source_points[:2] / source_points[2:].clamp_min(1e-6)

    source_x = source_points[0].reshape(out_h, out_w)
    source_y = source_points[1].reshape(out_h, out_w)
    norm_x = (source_x / max(tensor.shape[-1] - 1, 1)) * 2.0 - 1.0
    norm_y = (source_y / max(tensor.shape[-2] - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

    warped = F.grid_sample(tensor, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    warped = warped.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    return warped
