"""PyTorch Dataset for GTA-UAV supporting single-frame and windowed sequences.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import load_split, LocalizationSample
from .warp import OverheadWarpConfig, warp_to_overhead


def default_image_loader(p: Path):
    return Image.open(p).convert("RGB")


class GTAUAVDataset(Dataset):
    def __init__(
        self,
        split_json: str,
        dataset_root: str,
        mode: str = "single",
        window_size: int = 3,
        stride: int = 1,
        transform=None,
        loader=default_image_loader,
        sequences_cache: Optional[List[List[LocalizationSample]]] = None,
        warp_overhead: bool = False,
        warp_config: OverheadWarpConfig | None = None,
        warp_backend: str = "auto",
    ):
        self.samples = load_split(split_json, dataset_root)
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.loader = loader
        self.warp_overhead = warp_overhead
        self.warp_config = warp_config or OverheadWarpConfig()
        self.warp_backend = warp_backend

        # Build sequence windows if requested
        if self.mode.startswith("seq"):
            from .sequences import group_into_sequences, sliding_windows

            sequences = group_into_sequences(self.samples)
            self.windows = []
            for seq in sequences:
                for w in sliding_windows(seq, self.window_size, self.stride):
                    self.windows.append(w)
        else:
            self.windows = None

    def __len__(self):
        if self.windows is not None:
            return len(self.windows)
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.windows is not None:
            window = self.windows[idx]
            imgs = []
            metas = []
            for s in window:
                img_path: Path = s.drone_img_path
                img = self.loader(img_path)
                if self.warp_overhead:
                    img = warp_to_overhead(img, s.drone_metadata, self._warp_config_for_sample())
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
                positive_names = list(s.pair_pos_sate_img_list) + list(s.pair_pos_semipos_sate_img_list)
                positive_weights = list(s.pair_pos_sate_weight_list) + list(s.pair_pos_semipos_sate_weight_list)
                tile_id = positive_names[0] if positive_names else None
                metas.append(
                    {
                        "drone_img_path": str(s.drone_img_path),
                        "satellite_dir": str(s.satellite_dir),
                        "tile_id": tile_id,
                        "xy": s.drone_loc_x_y,
                        "positive_names": positive_names,
                        "positive_weights": positive_weights,
                    }
                )
            # Stack images along time dim
            imgs = torch.stack(imgs)
            return {"images": imgs, "meta": metas}
        else:
            s = self.samples[idx]
            img = self.loader(s.drone_img_path)
            if self.warp_overhead:
                img = warp_to_overhead(img, s.drone_metadata, self._warp_config_for_sample())
            if self.transform:
                img = self.transform(img)
            positive_names = list(s.pair_pos_sate_img_list) + list(s.pair_pos_semipos_sate_img_list)
            positive_weights = list(s.pair_pos_sate_weight_list) + list(s.pair_pos_semipos_sate_weight_list)
            tile_id = positive_names[0] if positive_names else None
            return {
                "image": img,
                "meta": {
                    "drone_img_path": str(s.drone_img_path),
                    "satellite_dir": str(s.satellite_dir),
                    "pose": s.drone_metadata,
                    "tile_id": tile_id,
                    "xy": s.drone_loc_x_y,
                    "positive_names": positive_names,
                    "positive_weights": positive_weights,
                },
            }

    def _warp_config_for_sample(self) -> OverheadWarpConfig:
        if self.warp_backend == "auto":
            backend = "torch" if torch.cuda.is_available() else "cpu"
        else:
            backend = self.warp_backend

        device = "cuda" if backend == "torch" and torch.cuda.is_available() else None
        return OverheadWarpConfig(
            output_size=self.warp_config.output_size,
            reference_height=self.warp_config.reference_height,
            field_of_view_degrees=self.warp_config.field_of_view_degrees,
            use_camera_pose=self.warp_config.use_camera_pose,
            backend=backend,
            device=device,
        )
