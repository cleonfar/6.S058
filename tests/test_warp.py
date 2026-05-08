from pathlib import Path

import numpy as np
from PIL import Image

from gtauav_loc import OverheadWarpConfig, warp_to_overhead


def test_warp_to_overhead_returns_expected_size_for_pil_input() -> None:
    image = Image.fromarray(np.full((64, 96, 3), 180, dtype=np.uint8))
    pose = {"cam_roll": -90.0, "cam_pitch": 0.0, "cam_yaw": 15.0, "height": 300.0}

    warped = warp_to_overhead(image, pose, OverheadWarpConfig(output_size=(128, 128)))

    assert isinstance(warped, Image.Image)
    assert warped.size == (128, 128)


def test_warp_to_overhead_accepts_numpy_input() -> None:
    image = np.full((64, 96, 3), 127, dtype=np.uint8)
    pose = {"drone_roll": -2.0, "drone_pitch": 1.0, "drone_yaw": 120.0, "height": 450.0}

    warped = warp_to_overhead(image, pose, OverheadWarpConfig(output_size=(80, 80)))

    assert isinstance(warped, np.ndarray)
    assert warped.shape == (80, 80, 3)


def test_warp_to_overhead_torch_backend_runs_on_cpu() -> None:
    image = np.full((48, 72, 3), 200, dtype=np.uint8)
    pose = {"cam_roll": -90.0, "cam_pitch": 0.0, "cam_yaw": 0.0, "height": 320.0}

    warped = warp_to_overhead(image, pose, OverheadWarpConfig(output_size=(64, 64), backend="torch", device="cpu"))

    assert isinstance(warped, np.ndarray)
    assert warped.shape == (64, 64, 3)
