"""Precompute satellite tile features and build an index for retrieval.

Usage:
    python -m gtauav_loc.satellite_precompute --dataset-root dataset --out-dir data/precompute --batch-size 64

This script:

If you only need the embeddings for training and evaluation, pass --skip-sift to
avoid the expensive CPU-side keypoint extraction step and keep the GPU busier.

Progress bars and timing are provided via tqdm.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from .local_features import extract_sift_features
from .utils.device import get_device


_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD = [0.229, 0.224, 0.225]


def list_tiles(dataset_root: str):
    tiles_dir = Path(dataset_root) / "satellite" / "satellite"
    files = sorted([p for p in tiles_dir.glob("**/*.png")])
    return files


def make_backbone(device):
    import torchvision.models as models
    from torchvision.models import ResNet18_Weights

    # Use ResNet18 so the satellite precompute matches the query encoder family.
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def make_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGE_MEAN, std=_IMAGE_STD),
    ])


def compute_embeddings(model, pil_images, transform, device):
    batch = torch.stack([transform(image) for image in pil_images]).to(device, non_blocking=True)
    with torch.inference_mode():
        embeddings = model(batch).detach().cpu().numpy().astype(np.float32)
    return embeddings


def save_npz(out_path: Path, emb, kps=None, desc=None):
    if kps is None:
        kps_array = np.zeros((0, 4), dtype=np.float32)
    else:
        kps_array = np.asarray(kps, dtype=np.float32)
        if kps_array.ndim == 1:
            kps_array = kps_array.reshape(1, -1)
    desc_array = np.asarray(desc, dtype=np.float32) if desc is not None else np.zeros((0, 128), dtype=np.float32)
    np.savez_compressed(out_path, emb=emb, kps=kps_array, desc=desc_array)


def build_faiss_index(embs, out_file: Path, use_gpu: bool = False):
    try:
        import faiss
    except Exception:
        print("FAISS is not installed; skipping index write.")
        return
    embs = np.vstack(embs).astype(np.float32)
    d = embs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embs)
    faiss.write_index(index, str(out_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--out-dir", default="data/precompute")
    # Backwards-compatible alias used by older scripts
    parser.add_argument("--precompute-dir", dest="out_dir", help="Alias for --out-dir (backwards compatibility)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--skip-sift", action="store_true", help="Skip local descriptor extraction and only save embeddings")
    parser.add_argument("--sift-target-num-features", type=int, default=8000, help="Requested SIFT feature count per tile for pypopsift")
    parser.add_argument("--sift-feature-process-size", type=int, default=2048, help="Max image size processed by pypopsift before internal resize")
    parser.add_argument("--sift-peak-threshold", type=float, default=0.1, help="SIFT peak threshold")
    parser.add_argument("--sift-edge-threshold", type=float, default=10.0, help="SIFT edge threshold")
    args = parser.parse_args()

    device = get_device(prefer_gpu=args.use_gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    tiles = list_tiles(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = make_backbone(device)
    transform = make_transform()

    embs = []
    files = []
    start = time.perf_counter()
    total_batches = max(1, (len(tiles) + args.batch_size - 1) // args.batch_size)
    empty_descriptor_tiles = 0
    total_sift_tiles = 0
    progress = tqdm(total=len(tiles), desc="Tiles", unit="tile")
    for batch_index in range(total_batches):
        batch_start = batch_index * args.batch_size
        batch_end = min(len(tiles), batch_start + args.batch_size)
        batch_paths = tiles[batch_start:batch_end]
        pil_images = []
        valid_paths = []
        for p in batch_paths:
            image = cv2.imread(str(p))
            if image is None:
                continue
            pil_images.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            valid_paths.append(p)

        if not valid_paths:
            progress.update(len(batch_paths))
            continue

        batch_embs = compute_embeddings(model, pil_images, transform, device)
        for p, emb, pil_image in zip(valid_paths, batch_embs, pil_images):
            if args.skip_sift:
                save_npz(out_dir / (p.stem + ".npz"), emb)
            else:
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                # Use the committed SuperPoint backend for local descriptors.
                features = extract_sift_features(
                    image,
                    backend="superpoint",
                    peak_threshold=args.sift_peak_threshold,
                    edge_threshold=args.sift_edge_threshold,
                    target_num_features=args.sift_target_num_features,
                    feature_process_size=args.sift_feature_process_size,
                )
                descriptor_count = 0 if features.descriptors is None else int(len(features.descriptors))
                total_sift_tiles += 1
                if descriptor_count == 0:
                    empty_descriptor_tiles += 1
                save_npz(out_dir / (p.stem + ".npz"), emb, features.keypoints, features.descriptors)
            embs.append(emb)
            files.append(str(p))

        progress.update(len(batch_paths))
        elapsed = time.perf_counter() - start
        processed = len(files)
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = len(tiles) - progress.n
        eta = remaining / rate if rate > 0 else 0.0
        progress.set_postfix_str(f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    progress.close()

    # Save index and mapping
    np.save(out_dir / "files.npy", np.array(files))
    build_faiss_index(embs, out_dir / "faiss.index", use_gpu=(device.type == "cuda"))

    took = time.perf_counter() - start
    if not args.skip_sift:
        print(
            f"Local-descriptor summary: empty_descriptors={empty_descriptor_tiles}/{max(1, total_sift_tiles)} "
            f"backend=superpoint target_num_features={args.sift_target_num_features}"
        )
    print(f"Precompute finished in {took:.1f}s — processed {len(files)} tiles")


if __name__ == "__main__":
    main()
