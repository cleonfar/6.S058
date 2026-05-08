#!/usr/bin/env python3
"""
Convert University-1652 dataset to the JSON format used by gtauav_loc trainers.

Input structure:
  University-Release/train/
    drone/<location_id>/image-01.jpeg, image-02.jpeg, ...
    satellite/<location_id>/<location_id>.jpg

Output JSON format:
  Each drone image becomes a sample paired to its location's satellite image.
  No coordinates, metadata, or semi-positives (University-1652 doesn't provide them).
"""

from pathlib import Path
import json
from typing import List, Dict, Any
import argparse


def convert_university_to_json(
    dataset_root: Path,
    split_name: str = "train",
    output_path: Path = None,
    train_frac: float = 0.9,
    seed: int = 42,
) -> None:
    """
    Convert University-1652 dataset to JSON format.
    
    Args:
        dataset_root: Path to University-Release folder
        split_name: "train" or "test"
        output_path: Where to write JSON (defaults to dataset_root/university_{split_name}.json)
        train_frac: Fraction of locations for training (only used if split_name="train")
        seed: Random seed for train/val split
    """
    dataset_root = Path(dataset_root)
    if output_path is None:
        output_path = dataset_root / f"university_{split_name}.json"
    
    if split_name == "train":
        samples = _convert_train_split(dataset_root, train_frac, seed)
    elif split_name == "test":
        samples = _convert_test_split(dataset_root)
    else:
        raise ValueError(f"Unknown split_name: {split_name}")
    
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"Wrote {len(samples)} samples to {output_path}")


def _convert_train_split(
    dataset_root: Path,
    train_frac: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """Convert the official train/ split into retrieval samples.

    The University-1652 release already provides a train folder and a test folder.
    We therefore export every training image here and leave train/val partitioning
    to the trainer.
    """
    drone_dir = dataset_root / "train" / "drone"
    satellite_dir = dataset_root / "train" / "satellite"
    
    if not drone_dir.exists() or not satellite_dir.exists():
        raise FileNotFoundError(f"train/drone or train/satellite not found in {dataset_root}")
    
    # Collect location IDs
    location_ids = sorted(set(p.name for p in drone_dir.iterdir() if p.is_dir()))
    print(f"Found {len(location_ids)} locations in train split")
    
    samples = []
    for loc_id in location_ids:
        drone_loc_dir = drone_dir / loc_id
        satellite_loc_dir = satellite_dir / loc_id

        if not satellite_loc_dir.exists():
            print(f"  Warning: satellite dir missing for location {loc_id}, skipping")
            continue

        # Find satellite image
        sat_images = list(satellite_loc_dir.glob("*.jpg")) + list(satellite_loc_dir.glob("*.jpeg"))
        if not sat_images:
            print(f"  Warning: no satellite image found for location {loc_id}, skipping")
            continue
        sat_image_name = sat_images[0].name

        # Each drone image in this location is a sample
        drone_images = sorted(list(drone_loc_dir.glob("*.jpg")) + list(drone_loc_dir.glob("*.jpeg")))
        for drone_image in drone_images:
            sample = {
                "drone_img_dir": "train/drone",
                "drone_img_name": f"{loc_id}/{drone_image.name}",
                "sate_img_dir": "train/satellite",
                "drone_loc_x_y": [0.0, 0.0],  # Not provided in University-1652
                "pair_pos_sate_img_list": [f"{loc_id}/{sat_image_name}"],
                "pair_pos_sate_weight_list": [1.0],
                "pair_pos_sate_loc_x_y_list": [[0.0, 0.0]],
                "pair_pos_semipos_sate_img_list": [],
                "pair_pos_semipos_sate_weight_list": [],
                "pair_pos_semipos_sate_loc_x_y_list": [],
                "drone_metadata": {},  # Not provided in University-1652
            }
            samples.append(sample)
    
    print(f"  Converted {len(samples)} drone-satellite pairs")
    return samples


def _convert_test_split(dataset_root: Path) -> List[Dict[str, Any]]:
    """
    Convert test/ split using query_drone as queries and gallery_satellite as gallery.
    
    For retrieval evaluation, each query_drone image is a sample that can match any
    gallery_satellite image from the same location.
    """
    query_drone_dir = dataset_root / "test" / "query_drone"
    gallery_sat_dir = dataset_root / "test" / "gallery_satellite"
    
    if not query_drone_dir.exists() or not gallery_sat_dir.exists():
        raise FileNotFoundError(f"test/query_drone or test/gallery_satellite not found in {dataset_root}")
    
    # Collect gallery locations
    gallery_locations = sorted(set(p.name for p in gallery_sat_dir.iterdir() if p.is_dir()))
    print(f"Found {len(gallery_locations)} gallery locations in test split")
    
    samples = []
    query_locations = sorted(set(p.name for p in query_drone_dir.iterdir() if p.is_dir()))
    
    for query_loc_id in query_locations:
        query_loc_dir = query_drone_dir / query_loc_id
        
        # All queries from this location match gallery_satellite/{query_loc_id}
        gallery_loc_dir = gallery_sat_dir / query_loc_id
        if not gallery_loc_dir.exists():
            print(f"  Warning: gallery location {query_loc_id} not found, skipping")
            continue
        
        sat_images = list(gallery_loc_dir.glob("*.jpg")) + list(gallery_loc_dir.glob("*.jpeg"))
        if not sat_images:
            print(f"  Warning: no satellite gallery image for location {query_loc_id}, skipping")
            continue
        
        # Each query drone image in this location is a sample
        query_images = sorted(list(query_loc_dir.glob("*.jpg")) + list(query_loc_dir.glob("*.jpeg")))
        for query_image in query_images:
            sample = {
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
            samples.append(sample)
    
    print(f"  Converted {len(samples)} test query samples")
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert University-1652 dataset to JSON format")
    parser.add_argument("--dataset-root", default="University-Release", help="Path to University-Release folder")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Which split to convert")
    parser.add_argument("--output", default=None, help="Output JSON path (default: dataset_root/university_{split}.json)")
    parser.add_argument("--train-frac", type=float, default=0.9, help="Fraction for train/val split (only for train split)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for location-based splits")
    args = parser.parse_args()
    
    convert_university_to_json(
        dataset_root=args.dataset_root,
        split_name=args.split,
        output_path=Path(args.output) if args.output else None,
        train_frac=args.train_frac,
        seed=args.seed,
    )
