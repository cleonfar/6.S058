"""Evaluation harness for GTA-UAV homography localization models.

Evaluates a trained checkpoint by running the same forward pass used during
training (drone + satellite backbone features -> HomographyHead -> predicted
corners) and reporting reprojection loss and corner error in pixels.

Two modes:
  --homography-dir   Fast path: load precomputed SIFT homographies from the
                     npz cache produced by homography_precompute.  Only samples
                     present in the cache are evaluated.

  (no --homography-dir)  Realistic path: compute SIFT homographies on-the-fly
                         for each test image, exactly as would happen at
                         deployment.  Slower (one SIFT match per sample) but
                         requires no precomputation on the query images.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from tqdm import tqdm

from .homography import HomographyHead, SpatialHomographyHead, compute_homography_reprojection_loss, estimate_homography_from_paths
from .homography_precompute import _resolve_satellite_path
from .torch_dataset import GTAUAVDataset
from .train_baseline import (
    QueryEncoder,
    SpatialEncoder,
    CorrelationVolume,
    OverheadWarpConfig,
    normalize_imagenet,
    load_homography_targets,
    load_sat_path_map,
    _load_image_as_tensor,
    normalize_path_key,
    resolve_valid_homography_indices,
)
from .utils.device import get_device
from .warp import batch_warp_to_overhead


@dataclass(frozen=True)
class EvaluationSummary:
    reproj_loss: float          # mean normalised L2 corner displacement (training loss)
    mean_corner_err_px: float   # same metric scaled to pixels (reproj_loss * warp_size)
    num_valid: int              # samples that had a valid homography target
    num_total: int              # total samples in the evaluated split
    on_the_fly: bool = False    # True when SIFT was computed on-the-fly (no precompute)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gtauav-eval",
        description="Evaluate a GTA-UAV homography checkpoint on reprojection loss",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--split", default="University-Release/university_train.json", help="Split JSON file")
    parser.add_argument("--dataset-root", default="University-Release", help="Dataset root directory")
    parser.add_argument(
        "--homography-dir",
        default=None,
        help=(
            "Directory containing homographies.npz (precomputed cache). "
            "If omitted, SIFT homographies are computed on-the-fly per sample "
            "(slower but requires no precomputation on query images)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--warp-size", type=int, default=224)
    parser.add_argument("--skip-warp", action="store_true", help="Disable overhead warp (must match training config)")
    parser.add_argument("--limit", type=int, default=0, help="Cap the number of samples evaluated (0 = all)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split=args.split,
        dataset_root=args.dataset_root,
        homography_dir=args.homography_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        warp_size=args.warp_size,
        skip_warp=args.skip_warp,
        limit=args.limit or None,
    )
    print(
        f"total={summary.num_total} valid_homographies={summary.num_valid} "
        f"reproj_loss={summary.reproj_loss:.4f} "
        f"mean_corner_err_px={summary.mean_corner_err_px:.2f}px"
        + (" [on-the-fly SIFT]" if summary.on_the_fly else " [precomputed]")
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split: str | Path,
    dataset_root: str | Path,
    homography_dir: str | Path | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    use_gpu: bool = False,
    warp_size: int = 224,
    skip_warp: bool = False,
    limit: int | None = None,
) -> EvaluationSummary:
    checkpoint_path = Path(checkpoint_path)
    device = get_device(prefer_gpu=use_gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------------------------- #
    # Load checkpoint: QueryEncoder backbone + HomographyHead
    # ---------------------------------------------------------------------- #
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]
    model_kind = checkpoint.get("model_kind", "")
    head_state = checkpoint.get("homography_head_state")
    if head_state is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not contain 'homography_head_state'. "
            "This checkpoint was not saved by the homography training script."
        )

    corr_state = checkpoint.get("corr_state")
    use_spatial = corr_state is not None or model_kind in (
        "spatial_crossattn_homography",
        "correlation_volume_homography",
    )

    if use_spatial:
        model = SpatialEncoder().to(device)
        model.load_state_dict(state_dict)
        model.eval()
        # CorrelationVolume has no parameters; saved corr_state is empty dict.
        corr = CorrelationVolume().to(device)
        corr.eval()
        _map_size = warp_size // 16
        _corr_channels = _map_size * _map_size  # 196 at 224px
        homography_head = SpatialHomographyHead(in_channels=_corr_channels).to(device)
    else:
        # Legacy global-pooling architecture
        _proj_w = state_dict.get("projection.2.weight")
        output_dim = int(_proj_w.shape[0]) if _proj_w is not None else 128
        model = QueryEncoder(output_dim=output_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        corr = None
        homography_head = HomographyHead(in_features=2 * model.feature_dim).to(device)

    homography_head.load_state_dict(head_state)
    homography_head.eval()

    # ---------------------------------------------------------------------- #
    # Load homography targets (precomputed fast path) or prepare on-the-fly
    # ---------------------------------------------------------------------- #
    on_the_fly = homography_dir is None
    if not on_the_fly:
        homography_targets = load_homography_targets(homography_dir, model_size=(warp_size, warp_size))
        sat_path_map = load_sat_path_map(homography_dir)
        print(f"Precomputed homography targets: {len(homography_targets)} valid entries")
    else:
        homography_targets = None
        sat_path_map = None
        print("On-the-fly SIFT mode: computing homographies per sample (no precompute required)")

    # ---------------------------------------------------------------------- #
    # Build dataset — full split in on-the-fly mode, valid subset otherwise
    # ---------------------------------------------------------------------- #
    dataset = GTAUAVDataset(
        split_json=split,
        dataset_root=dataset_root,
        mode="single",
        transform=T.Compose([T.Resize((warp_size, warp_size)), T.ToTensor()]),
        warp_overhead=False,
    )
    num_total = len(dataset)

    if not on_the_fly:
        valid_indices = resolve_valid_homography_indices(dataset, homography_dir)
        print(f"Dataset: {num_total} total, {len(valid_indices)} with valid precomputed homographies")
        if limit is not None:
            valid_indices = valid_indices[:limit]
        eval_dataset = Subset(dataset, valid_indices)
    else:
        print(f"Dataset: {num_total} total (all samples attempted in on-the-fly mode)")
        if limit is not None:
            eval_dataset = Subset(dataset, list(range(min(limit, num_total))))
        else:
            eval_dataset = dataset
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        collate_fn=_collate_batch,
    )

    warp_config = OverheadWarpConfig(
        output_size=(warp_size, warp_size),
        backend="skip" if skip_warp else "torch",
        device=str(device) if device.type == "cuda" else None,
    )

    # ---------------------------------------------------------------------- #
    # Evaluation loop (no gradients, mirrors run_epoch without backward)
    # ---------------------------------------------------------------------- #
    reproj_losses: List[float] = []
    num_valid_total = 0
    skipped = 0

    progress = tqdm(loader, desc="eval", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        metas = batch["meta"]
        poses = [meta["pose"] for meta in metas]

        valid_idx: List[int] = []
        target_h_list: List[np.ndarray] = []
        sat_tensors: List[torch.Tensor] = []

        for i, meta in enumerate(metas):
            if on_the_fly:
                # Resolve satellite path from metadata (no precomputed map needed).
                sat_dir = Path(meta.get("satellite_dir", ""))
                positive_names = meta.get("positive_names") or []
                if not positive_names:
                    continue
                sat_path = _resolve_satellite_path(sat_dir, positive_names[0])
                if not sat_path.exists():
                    continue
                drone_path = Path(meta.get("drone_img_path", ""))
                if not drone_path.exists():
                    continue
                H = estimate_homography_from_paths(drone_path, sat_path, sift_backend="opencv")
                if H is None:
                    continue
                # Rescale H from original pixel space to model space.
                drone_img = cv2.imread(str(drone_path))
                sat_img = cv2.imread(str(sat_path))
                if drone_img is None or sat_img is None:
                    continue
                h0, w0 = drone_img.shape[:2]
                hs0, ws0 = sat_img.shape[:2]
                A_d = np.array([[w0 / warp_size, 0, 0], [0, h0 / warp_size, 0], [0, 0, 1]], np.float32)
                A_s_inv = np.array([[warp_size / ws0, 0, 0], [0, warp_size / hs0, 0], [0, 0, 1]], np.float32)
                target_h = (A_s_inv @ H @ A_d).astype(np.float32)
            else:
                key = normalize_path_key(meta.get("drone_img_path", ""))
                target_h = homography_targets.get(key)
                if target_h is None:
                    continue
                sat_path = sat_path_map.get(key)
                if sat_path is None:
                    continue
                sat_path = Path(sat_path)

            sat_t = _load_image_as_tensor(str(sat_path), warp_size)
            if sat_t is None:
                continue
            valid_idx.append(i)
            target_h_list.append(target_h)
            sat_tensors.append(sat_t)

        if not valid_idx:
            skipped += 1
            continue

        sat_images = torch.cat(sat_tensors, dim=0).to(device, non_blocking=True)

        with torch.no_grad():
            warped = batch_warp_to_overhead(images, poses, warp_config) if warp_config.backend == "torch" else images
            norm_drone = normalize_imagenet(warped)
            norm_sat = normalize_imagenet(sat_images)

            idx_t = torch.tensor(valid_idx, dtype=torch.long, device=device)
            if use_spatial:
                drone_maps = model(torch.index_select(norm_drone, 0, idx_t))
                sat_maps = model(norm_sat)
                combined = corr(drone_maps, sat_maps)
            else:
                drone_feats = model.backbone(torch.index_select(norm_drone, 0, idx_t))
                sat_feats = model.backbone(norm_sat)
                combined = torch.cat([drone_feats, sat_feats], dim=1)

            pred_corners = homography_head(combined)

            target_h_t = torch.from_numpy(np.stack(target_h_list)).to(device=device, dtype=pred_corners.dtype)
            loss = compute_homography_reprojection_loss(
                pred_corners, target_h_t, image_size=(warp_size, warp_size)
            )

        reproj_losses.append(float(loss.cpu()))
        num_valid_total += len(valid_idx)
        progress.set_postfix(loss=f"{reproj_losses[-1]:.4f}")

    if skipped:
        print(f"  [info] {skipped} batch(es) had no samples with a recoverable homography")

    if not reproj_losses:
        raise RuntimeError("No valid homography samples found in the evaluated split.")

    mean_reproj = float(np.mean(reproj_losses))
    mean_corner_px = mean_reproj * warp_size

    return EvaluationSummary(
        reproj_loss=mean_reproj,
        mean_corner_err_px=mean_corner_px,
        num_valid=num_valid_total,
        num_total=num_total,
        on_the_fly=on_the_fly,
    )


def _collate_batch(batch):
    images = torch.stack([item["image"] for item in batch])
    metas = [item["meta"] for item in batch]
    return {"image": images, "meta": metas}


if __name__ == "__main__":
    main()
