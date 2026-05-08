from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import json
from pathlib import PurePath

import warnings


@dataclass(frozen=True)
class LocalizationSample:
    drone_img_path: Path
    satellite_dir: Path
    drone_loc_x_y: Tuple[float, float]
    pair_pos_sate_img_list: Tuple[str, ...]
    pair_pos_sate_weight_list: Tuple[float, ...]
    pair_pos_sate_loc_x_y_list: Tuple[Tuple[float, float], ...]
    pair_pos_semipos_sate_img_list: Tuple[str, ...]
    pair_pos_semipos_sate_weight_list: Tuple[float, ...]
    pair_pos_semipos_sate_loc_x_y_list: Tuple[Tuple[float, float], ...]
    drone_metadata: Dict[str, float]


def load_split(json_path: str | Path, dataset_root: str | Path) -> List[LocalizationSample]:
    root = Path(dataset_root)
    split_path = Path(json_path)
    if not split_path.is_file():
        candidate = root / split_path
        if candidate.is_file():
            split_path = candidate
    # If no JSON found, attempt to auto-discover University-style layout
    if not split_path.is_file():
        # Determine requested split name from provided path
        split_name = PurePath(split_path).stem.lower()
        if split_name not in ("train", "test"):
            # default to train if unspecified
            split_name = "train"
        # Check for University-1652 style layout
        train_dir = root / "train"
        test_dir = root / "test"
        if train_dir.is_dir() and test_dir.is_dir():
            # Build samples from directory layout similar to convert_university_to_json.py
            raw_items = _discover_university_split(root, split_name)
        else:
            raise FileNotFoundError(f"Split JSON not found: {json_path} and no recognizable dataset layout under {dataset_root}")
    else:
        with split_path.open("r", encoding="utf-8") as handle:
            raw_items = json.load(handle)

    samples: List[LocalizationSample] = []
    for item in raw_items:
        samples.append(
            LocalizationSample(
                drone_img_path=root / item["drone_img_dir"] / item["drone_img_name"],
                satellite_dir=root / item["sate_img_dir"],
                drone_loc_x_y=_as_xy_tuple(item["drone_loc_x_y"]),
                pair_pos_sate_img_list=tuple(item.get("pair_pos_sate_img_list", [])),
                pair_pos_sate_weight_list=tuple(float(value) for value in item.get("pair_pos_sate_weight_list", [])),
                pair_pos_sate_loc_x_y_list=tuple(_as_xy_tuple(value) for value in item.get("pair_pos_sate_loc_x_y_list", [])),
                pair_pos_semipos_sate_img_list=tuple(item.get("pair_pos_semipos_sate_img_list", [])),
                pair_pos_semipos_sate_weight_list=tuple(float(value) for value in item.get("pair_pos_semipos_sate_weight_list", [])),
                pair_pos_semipos_sate_loc_x_y_list=tuple(_as_xy_tuple(value) for value in item.get("pair_pos_semipos_sate_loc_x_y_list", [])),
                drone_metadata={key: float(value) for key, value in item.get("drone_metadata", {}).items()},
            )
        )
    return samples

def _discover_university_split(dataset_root: Path, split: str) -> List[dict]:
    """Discover University-1652 style samples in-memory.

    Returns a list of dicts compatible with the JSON schema expected by the loader.
    """
    samples = []
    if split == "train":
        drone_dir = dataset_root / "train" / "drone"
        satellite_dir = dataset_root / "train" / "satellite"
        if not drone_dir.exists() or not satellite_dir.exists():
            raise FileNotFoundError(f"train/drone or train/satellite not found in {dataset_root}")

        location_ids = sorted(p.name for p in drone_dir.iterdir() if p.is_dir())
        for loc_id in location_ids:
            drone_loc_dir = drone_dir / loc_id
            satellite_loc_dir = satellite_dir / loc_id
            if not satellite_loc_dir.exists():
                warnings.warn(f"satellite dir missing for location {loc_id}, skipping")
                continue
            sat_images = list(satellite_loc_dir.glob("*.jpg")) + list(satellite_loc_dir.glob("*.jpeg"))
            if not sat_images:
                warnings.warn(f"no satellite image found for location {loc_id}, skipping")
                continue
            sat_image_name = sat_images[0].name
            drone_images = sorted(list(drone_loc_dir.glob("*.jpg")) + list(drone_loc_dir.glob("*.jpeg")))
            for drone_image in drone_images:
                samples.append(
                    {
                        "drone_img_dir": "train/drone",
                        "drone_img_name": f"{loc_id}/{drone_image.name}",
                        "sate_img_dir": "train/satellite",
                        "drone_loc_x_y": [0.0, 0.0],
                        "pair_pos_sate_img_list": [f"{loc_id}/{sat_image_name}"],
                        "pair_pos_sate_weight_list": [1.0],
                        "pair_pos_sate_loc_x_y_list": [[0.0, 0.0]],
                        "pair_pos_semipos_sate_img_list": [],
                        "pair_pos_semipos_sate_weight_list": [],
                        "pair_pos_semipos_sate_loc_x_y_list": [],
                        "drone_metadata": {},
                    }
                )
    else:
        # test split: query_drone / gallery_satellite
        query_drone_dir = dataset_root / "test" / "query_drone"
        gallery_sat_dir = dataset_root / "test" / "gallery_satellite"
        if not query_drone_dir.exists() or not gallery_sat_dir.exists():
            raise FileNotFoundError(f"test/query_drone or test/gallery_satellite not found in {dataset_root}")
        query_locations = sorted(p.name for p in query_drone_dir.iterdir() if p.is_dir())
        for query_loc_id in query_locations:
            query_loc_dir = query_drone_dir / query_loc_id
            gallery_loc_dir = gallery_sat_dir / query_loc_id
            if not gallery_loc_dir.exists():
                warnings.warn(f"gallery location {query_loc_id} not found, skipping")
                continue
            sat_images = list(gallery_loc_dir.glob("*.jpg")) + list(gallery_loc_dir.glob("*.jpeg"))
            if not sat_images:
                warnings.warn(f"no satellite gallery image for location {query_loc_id}, skipping")
                continue
            query_images = sorted(list(query_loc_dir.glob("*.jpg")) + list(query_loc_dir.glob("*.jpeg")))
            for query_image in query_images:
                samples.append(
                    {
                        "drone_img_dir": "test/query_drone",
                        "drone_img_name": f"{query_loc_id}/{query_image.name}",
                        "sate_img_dir": "test/gallery_satellite",
                        "drone_loc_x_y": [0.0, 0.0],
                        "pair_pos_sate_img_list": [f"{query_loc_id}/{sat_image_name}"],
                        "pair_pos_sate_weight_list": [1.0],
                        "pair_pos_sate_loc_x_y_list": [[0.0, 0.0]],
                        "pair_pos_semipos_sate_img_list": [],
                        "pair_pos_semipos_sate_weight_list": [],
                        "pair_pos_semipos_sate_loc_x_y_list": [],
                        "drone_metadata": {},
                    }
                )
    return samples

def _as_xy_tuple(value: Sequence[Any]) -> Tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"Expected an x/y pair, got: {value}")
    return float(value[0]), float(value[1])
