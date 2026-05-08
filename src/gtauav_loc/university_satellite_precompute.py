#!/usr/bin/env python3
"""
Precompute satellite embeddings for University-1652 dataset.

Usage:
    python -m gtauav_loc.university_satellite_precompute --dataset-root University-Release --out-dir data/precompute_university --split train

This generates .npz files with embeddings indexed by location ID.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from .utils.device import get_device


_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD = [0.229, 0.224, 0.225]


def list_satellite_locations(dataset_root: Path, split: str) -> List[tuple[str, Path]]:
    """
    List all satellite images and their location IDs.
    
    Returns list of (location_id, satellite_image_path) tuples.
    """
    if split == "train":
        sat_dir = dataset_root / "train" / "satellite"
    elif split == "test":
        sat_dir = dataset_root / "test" / "gallery_satellite"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    locations = []
    for loc_dir in sorted(sat_dir.iterdir()):
        if not loc_dir.is_dir():
            continue
        sat_images = list(loc_dir.glob("*.jpg")) + list(loc_dir.glob("*.jpeg"))
        if sat_images:
            locations.append((loc_dir.name, sat_images[0]))
    
    return locations


def make_backbone(device, checkpoint_path: str | None = None):
    """Create the embedding model for satellite images.

    If *checkpoint_path* is given, loads the trained QueryEncoder from that
    checkpoint so the gallery embeddings live in the same space as drone query
    embeddings produced by evaluate.py.  Otherwise falls back to a pretrained
    ResNet18 backbone (used for the initial train-split precompute).
    """
    if checkpoint_path is not None:
        # Import here to avoid circular imports at module level.
        from .train_baseline import QueryEncoder
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_args = ckpt.get("args", {})
        output_dim = int(saved_args.get("output_dim", 128))
        full_model = QueryEncoder(output_dim=output_dim)
        full_model.load_state_dict(ckpt["model_state"])
        # Use the backbone only — the projection head was never trained with a
        # retrieval loss, so its 128-dim output carries no semantic meaning for
        # retrieval.  The backbone (512-dim) received gradients from the
        # homography head and produces meaningful features.
        backbone = full_model.backbone
        feature_dim = full_model.feature_dim
        backbone.eval()
        backbone.to(device)
        print(f"Loaded trained backbone (feature_dim={feature_dim}) from {checkpoint_path}")
        return backbone, feature_dim

    import torchvision.models as models
    from torchvision.models import ResNet18_Weights

    backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_dim = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    backbone.to(device)
    return backbone, feature_dim


def make_transform():
    """Create image preprocessing pipeline."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGE_MEAN, std=_IMAGE_STD),
    ])


def compute_embeddings(model, pil_images, transform, device):
    """Compute embeddings for a batch of PIL images."""
    batch = torch.stack([transform(image) for image in pil_images]).to(device, non_blocking=True)
    with torch.inference_mode():
        out = model(batch)
        # QueryEncoder returns normalized embeddings; plain backbone returns raw features.
        import torch.nn.functional as F
        embeddings = F.normalize(out, dim=-1).detach().cpu().numpy().astype(np.float32)
    return embeddings


def save_npz(out_path: Path, emb: np.ndarray, kps=None, desc=None):
    """Save embedding and optional features to .npz file."""
    if kps is None:
        kps_array = np.zeros((0, 4), dtype=np.float32)
    else:
        kps_array = np.asarray(kps, dtype=np.float32)
        if kps_array.ndim == 1:
            kps_array = kps_array.reshape(1, -1)
    desc_array = np.asarray(desc, dtype=np.float32) if desc is not None else np.zeros((0, 128), dtype=np.float32)
    np.savez_compressed(out_path, emb=emb, kps=kps_array, desc=desc_array)


def main():
    parser = argparse.ArgumentParser(description="Precompute satellite embeddings for University-1652")
    parser.add_argument("--dataset-root", default="University-Release", help="Path to University-Release folder")
    parser.add_argument("--out-dir", default="data/precompute_university", help="Output directory for embeddings")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Which split to precompute")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding computation")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for computation")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained checkpoint (.pt). When provided the gallery is embedded "
             "with that model so embeddings are in the same space as evaluate.py queries. "
             "Required for meaningful retrieval evaluation; omit only for the initial "
             "train-split precompute with the pretrained baseline.",
    )
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device(prefer_gpu=args.use_gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    print(f"Computing embeddings for University-1652 {args.split} split...")
    print(f"Device: {device}")
    
    # List satellite images
    locations = list_satellite_locations(dataset_root, args.split)
    print(f"Found {len(locations)} satellite images")
    
    if not locations:
        print("No satellite images found!")
        return
    
    # Load model
    model, _emb_dim = make_backbone(device, checkpoint_path=args.checkpoint)
    transform = make_transform()
    
    # Process in batches
    batch_size = args.batch_size
    embeddings_list = []
    filenames = []
    
    with tqdm(total=len(locations), desc="Computing embeddings") as pbar:
        for i in range(0, len(locations), batch_size):
            batch_locations = locations[i:i+batch_size]
            batch_images = []
            batch_loc_ids = []
            
            for loc_id, sat_path in batch_locations:
                try:
                    img = Image.open(sat_path).convert("RGB")
                    batch_images.append(img)
                    batch_loc_ids.append(loc_id)
                except Exception as e:
                    print(f"  Warning: failed to load {sat_path}: {e}")
            
            if batch_images:
                embeddings = compute_embeddings(model, batch_images, transform, device)
                for loc_id, emb in zip(batch_loc_ids, embeddings):
                    npz_path = out_dir / f"{loc_id}.npz"
                    save_npz(npz_path, emb.reshape(-1))
                    filenames.append(f"{loc_id}.jpg")
                    embeddings_list.append(emb)
            
            pbar.update(len(batch_locations))
    
    # Save file list
    file_list = np.array(filenames, dtype=object)
    np.save(out_dir / "files.npy", file_list)
    
    print(f"Saved {len(embeddings_list)} embeddings to {out_dir}")
    print(f"File list saved to {out_dir}/files.npy")


if __name__ == "__main__":
    main()
