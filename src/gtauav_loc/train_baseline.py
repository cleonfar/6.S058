"""Baseline homography training script using reprojection loss.

This trainer learns a homography regressor on top of a CNN backbone and uses
only homography reprojection error as the optimization objective.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from .data import LocalizationSample
from .homography import HomographyHead, SpatialHomographyHead, compute_homography_reprojection_loss
import cv2
from .torch_dataset import GTAUAVDataset
from .utils.device import get_device
from .warp import OverheadWarpConfig, batch_warp_to_overhead


DEFAULT_TOP_K = (1, 5, 10)


@dataclass(frozen=True)
class EpochStats:
    loss: float
    recall_at_k: dict[int, float] | None = None
    localization_error: float | None = None
    reprojection_loss: float | None = None


class SpatialEncoder(nn.Module):
    """ResNet18 encoder that returns spatial feature maps from layer3.

    Stops before the global average pool so spatial structure is preserved.
    Output shape: (B, 256, H/16, W/16)  — at 224px input: (B, 256, 14, 14).
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Keep everything up to and including layer3; drop layer4, avgpool, fc.
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        self.feature_channels = 256  # layer3 output channels for ResNet18

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)  # (B, 256, H/16, W/16)


class CorrelationVolume(nn.Module):
    """Dense feature correlation between drone and satellite spatial maps.

    Computes a full pairwise cosine-similarity matrix between every drone
    feature position and every satellite feature position.  No learnable
    parameters — the signal is purely the visual similarity between patches,
    which generalises trivially to unseen satellite tiles.

    Input  : drone_map (B, C, Hd, Wd), sat_map (B, C, Hs, Ws)
    Output : (B, Hd*Wd, Hs, Ws)  correlation volume
             Entry [b, i, y, x] = cosine-sim between drone position i and
             satellite position (y,x).  Passed to SpatialHomographyHead.
    """

    def forward(self, drone_map: torch.Tensor, sat_map: torch.Tensor) -> torch.Tensor:
        B, C, Hd, Wd = drone_map.shape
        _, _, Hs, Ws = sat_map.shape
        Nd = Hd * Wd

        # L2-normalise along channel dim so dot product = cosine similarity
        d = F.normalize(drone_map.flatten(2), dim=1)  # (B, C, Nd)
        s = F.normalize(sat_map.flatten(2),   dim=1)  # (B, C, Ns)

        # (B, Nd, Ns) cosine similarity matrix
        corr = torch.bmm(d.permute(0, 2, 1), s)  # (B, Nd, Ns)

        # Reshape satellite axis back to spatial: (B, Nd, Hs, Ws)
        return corr.view(B, Nd, Hs, Ws)


# Legacy alias kept for evaluate.py checkpoint loading compatibility.
class QueryEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=-1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gtauav-train-baseline", description="Train homography alignment with reprojection loss")
    parser.add_argument("--split", default="dataset/same-area-drone2sate-train.json")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--precompute-dir", default="data/precompute_res18_sift")
    parser.add_argument("--out-dir", default="runs/baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--warp-backend", choices=["auto", "cpu", "torch"], default="auto")
    parser.add_argument("--warp-size", type=int, default=224)
    parser.add_argument("--skip-warp", action="store_true", help="Disable overhead warping preprocessing")
    parser.add_argument("--window-size", type=int, default=8, help="Window size to use for baseline training (uses all frames in window)")
    parser.add_argument("--stride", type=int, default=1, help="Window stride for baseline training")
    parser.add_argument("--mode", choices=["single", "seq"], default="single", help="Training mode (reprojection training supports single mode only)")
    parser.add_argument(
        "--homography-dir",
        "--homography-precompute",
        dest="homography_dir",
        required=True,
        help="Directory containing homographies.npz from homography_precompute",
    )
    parser.add_argument("--homography-loss-weight", type=float, default=1.0, help="Weight for homography reprojection loss")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = get_device(prefer_gpu=args.use_gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode != "single":
        raise ValueError("Reprojection-only training supports only --mode single")

    dataset = GTAUAVDataset(
        split_json=args.split,
        dataset_root=args.dataset_root,
        mode="single",
        transform=T.Compose([T.Resize((args.warp_size, args.warp_size)), T.ToTensor()]),
        warp_overhead=False,
        warp_config=OverheadWarpConfig(output_size=(args.warp_size, args.warp_size), backend=args.warp_backend),
        warp_backend=args.warp_backend,
    )
    collate_fn = collate_batch

    valid_indices = resolve_valid_homography_indices(dataset, args.homography_dir)
    dataset = Subset(dataset, valid_indices)
    homography_targets = load_homography_targets(args.homography_dir, model_size=(args.warp_size, args.warp_size))
    sat_path_map = load_sat_path_map(args.homography_dir)
    print(f"Using {len(valid_indices)} samples with valid homographies from {args.homography_dir}")

    train_indices, val_indices = split_indices(len(dataset), args.val_fraction, args.seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0, drop_last=False, collate_fn=collate_fn) if val_indices else None

    model = SpatialEncoder().to(device)
    # CorrelationVolume has no learnable params — it outputs (B, Nd, Hs, Ws)
    # where Nd = (warp_size//16)^2 = 196 at 224px.
    _map_size = args.warp_size // 16  # 14 at 224px
    _corr_channels = _map_size * _map_size  # 196 — one channel per drone position
    corr = CorrelationVolume().to(device)
    homography_head = SpatialHomographyHead(in_channels=_corr_channels).to(device)
    if args.freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    trainable_parameters = list(filter(lambda parameter: parameter.requires_grad, model.parameters()))
    trainable_parameters += list(corr.parameters())
    if homography_head is not None:
        trainable_parameters.extend(homography_head.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val = math.inf
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            corr=corr,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            train=True,
            warp_config=OverheadWarpConfig(output_size=(args.warp_size, args.warp_size), backend="torch" if not args.skip_warp else "skip", device=str(device) if device.type == "cuda" else None),
            homography_targets=homography_targets,
            homography_head=homography_head,
            homography_loss_weight=args.homography_loss_weight,
            epoch=epoch,
            total_epochs=args.epochs,
            sat_path_map=sat_path_map,
            warp_size=args.warp_size,
        )

        val_loss = None
        if val_loader is not None:
            val_loss = run_epoch(
                model=model,
                corr=corr,
                loader=val_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                train=False,
                warp_config=OverheadWarpConfig(output_size=(args.warp_size, args.warp_size), backend="torch" if not args.skip_warp else "skip", device=str(device) if device.type == "cuda" else None),
                homography_targets=homography_targets,
                homography_head=homography_head,
                homography_loss_weight=args.homography_loss_weight,
                epoch=epoch,
                total_epochs=args.epochs,
                sat_path_map=sat_path_map,
                warp_size=args.warp_size,
            )

        summary = f"epoch={epoch}/{args.epochs} train_loss={train_loss.loss:.4f}"
        if train_loss.reprojection_loss is not None:
            summary += f" train_reproj={train_loss.reprojection_loss:.4f}"
        if val_loss is not None:
            summary += f" val_loss={val_loss.loss:.4f}"
            if val_loss.reprojection_loss is not None:
                summary += f" val_reproj={val_loss.reprojection_loss:.4f}"
        summary += f" amp_scale={scaler.get_scale():.0f}"
        summary += f" elapsed={time.perf_counter() - start_time:.1f}s"
        print(summary)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "corr_state": corr.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "model_kind": "correlation_volume_homography",
        }
        if homography_head is not None:
            checkpoint["homography_head_state"] = homography_head.state_dict()
        torch.save(checkpoint, out_dir / "last.pt")
        if val_loss is not None and val_loss.loss < best_val:
            best_val = val_loss.loss
            torch.save(checkpoint, out_dir / "best.pt")

    print(f"Training finished in {time.perf_counter() - start_time:.1f}s")


def run_epoch(
    model: SpatialEncoder,
    corr: CorrelationVolume,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    train: bool,
    warp_config: OverheadWarpConfig,
    homography_targets: dict[str, np.ndarray] | None,
    homography_head: SpatialHomographyHead | None,
    homography_loss_weight: float,
    epoch: int,
    total_epochs: int,
    sat_path_map: dict[str, str] | None = None,
    warp_size: int = 224,
) -> EpochStats:
    model.train(train)
    corr.train(train)
    if homography_head is not None:
        homography_head.train(train)
    losses: List[float] = []
    reprojection_losses: List[float] = []
    progress = tqdm(loader, desc=f"{'train' if train else 'val'} {epoch}/{total_epochs}", leave=False)
    for batch in progress:
        if "image" not in batch:
            raise ValueError("Reprojection-only training expects single-frame batches")

        images = batch["image"].to(device, non_blocking=True)
        metas = batch["meta"]
        poses = [meta["pose"] for meta in metas]

        valid_idx: List[int] = []
        target_h_list: List[np.ndarray] = []
        sat_tensors: List[torch.Tensor] = []

        for i, meta in enumerate(metas):
            key = normalize_path_key(meta.get("drone_img_path", ""))
            target_h = homography_targets.get(key) if homography_targets is not None else None
            if target_h is None:
                continue
            sat_path = sat_path_map.get(key) if sat_path_map else None
            if sat_path is None:
                continue
            sat_t = _load_image_as_tensor(sat_path, warp_size)
            if sat_t is None:
                continue
            valid_idx.append(i)
            target_h_list.append(target_h)
            sat_tensors.append(sat_t)

        if not valid_idx:
            continue

        sat_images = torch.cat(sat_tensors, dim=0).to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            warped = batch_warp_to_overhead(images, poses, warp_config) if warp_config.backend == "torch" else images

            # Photometric augmentation only during training; identical jitter
            # is NOT shared across drone/sat (they should learn invariance to
            # lighting independently).
            if train:
                idx_t = torch.tensor(valid_idx, dtype=torch.long, device=device)
                drone_in = torch.index_select(warped, 0, idx_t)
                drone_in = photometric_augment(drone_in)
                sat_in = photometric_augment(sat_images)
            else:
                idx_t = torch.tensor(valid_idx, dtype=torch.long, device=device)
                drone_in = torch.index_select(warped, 0, idx_t)
                sat_in = sat_images

            norm_drone = normalize_imagenet(drone_in)
            norm_sat = normalize_imagenet(sat_in)

            drone_maps = model(norm_drone)                 # (N, C, H, W)
            sat_maps = model(norm_sat)                     # (N, C, H, W)
            combined = corr(drone_maps, sat_maps)          # (N, 2*C, H, W)

            pred_params = homography_head(combined)
            target_h_tensor = torch.from_numpy(np.stack(target_h_list)).to(device=device, dtype=pred_params.dtype)
            reprojection_loss = compute_homography_reprojection_loss(
                pred_params,
                target_h_tensor,
                image_size=(warp_size, warp_size),
            )
            loss = homography_loss_weight * reprojection_loss
            reprojection_losses.append(float(reprojection_loss.detach().cpu()))

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(corr.parameters()) + list(homography_head.parameters()),
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()

        losses.append(float(loss.detach().cpu()))
        progress.set_postfix(loss=f"{losses[-1]:.4f}")

    return EpochStats(
        loss=float(np.mean(losses)) if losses else math.inf,
        recall_at_k=None,
        localization_error=None,
        reprojection_loss=float(np.mean(reprojection_losses)) if reprojection_losses else None,
    )


def build_target_distribution(
    metas: Sequence[dict],
    store: SatelliteEmbeddingStore,
    device: torch.device,
    gallery_index_by_stem: dict[str, int],
) -> torch.Tensor:
    target = torch.zeros(len(metas), store.embeddings.shape[0], device=device, dtype=torch.float32)
    for row_index, meta in enumerate(metas):
        positive_names = list(meta.get("positive_names", []))
        positive_weights = list(meta.get("positive_weights", []))
        indices: List[int] = []
        weights: List[float] = []
        for pos_index, name in enumerate(positive_names):
            stem = Path(name).stem
            if stem not in gallery_index_by_stem:
                continue
            indices.append(gallery_index_by_stem[stem])
            if positive_weights and len(positive_weights) == len(positive_names):
                weights.append(float(positive_weights[pos_index]))
            else:
                weights.append(1.0)
        if not indices:
            raise KeyError(f"No positive satellite candidates available for sample {meta.get('drone_img_path', '<unknown>')}")
        weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
        weights_tensor = weights_tensor / weights_tensor.sum().clamp_min(1e-6)
        target[row_index, torch.tensor(indices, device=device, dtype=torch.long)] = weights_tensor
    return target


def build_window_target_distribution(
    metas: Sequence[Sequence[dict]],
    store: SatelliteEmbeddingStore,
    device: torch.device,
    gallery_index_by_stem: dict[str, int],
) -> torch.Tensor:
    rows = []
    for window_metas in metas:
        rows.append(build_target_distribution(window_metas, store, device, gallery_index_by_stem))
    return torch.stack(rows, dim=0)


def update_recall_metrics(recall_hits: dict[int, int], top_indices: torch.Tensor, metas: Sequence[dict], gallery_stems: Sequence[str]) -> None:
    for row, meta in zip(top_indices.tolist(), metas):
        gt_names = {Path(name).stem for name in meta.get("positive_names", [])}
        row_stems = [gallery_stems[index] for index in row]
        for k in DEFAULT_TOP_K:
            if any(stem in gt_names for stem in row_stems[:k]):
                recall_hits[k] += 1


def update_window_recall_metrics(recall_hits: dict[int, int], top_indices: torch.Tensor, metas: Sequence[Sequence[dict]], gallery_stems: Sequence[str]) -> None:
    batch_size = len(metas)
    seq_len = len(metas[0]) if batch_size else 0
    for b in range(batch_size):
        for t in range(seq_len):
            row = top_indices[b * seq_len + t].tolist()
            gt_names = {Path(name).stem for name in metas[b][t].get("positive_names", [])}
            row_stems = [gallery_stems[index] for index in row]
            for k in DEFAULT_TOP_K:
                if any(stem in gt_names for stem in row_stems[:k]):
                    recall_hits[k] += 1


def resolve_target_embedding(meta: dict, store: SatelliteEmbeddingStore) -> np.ndarray:
    positive_names = list(meta.get("positive_names", []))
    weights: Sequence[float] = list(meta.get("positive_weights", []))
    if not positive_names:
        raise KeyError("No positive satellite candidates available for this sample")
    return store.resolve_target(positive_names, weights)


def normalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def photometric_augment(images: torch.Tensor) -> torch.Tensor:
    """Apply random photometric jitter independently per sample.

    Input/output : (B, 3, H, W) float tensor in [0, 1] range.
    Modes applied (independent random choice per sample):
        * brightness scale  ~ U(0.7, 1.3)
        * contrast scale    ~ U(0.7, 1.3)
        * saturation scale  ~ U(0.7, 1.3)
        * gaussian noise    sigma ~ U(0.0, 0.03)
        * random grayscale  probability 0.10
    Output is clamped to [0, 1].
    """
    B = images.shape[0]
    device = images.device
    dtype = images.dtype

    def _u(low: float, high: float) -> torch.Tensor:
        return torch.empty(B, 1, 1, 1, device=device, dtype=dtype).uniform_(low, high)

    # Brightness
    images = images * _u(0.7, 1.3)
    # Contrast about per-image mean
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    images = (images - mean) * _u(0.7, 1.3) + mean
    # Saturation about per-pixel grayscale mean (across channels)
    gray = images.mean(dim=1, keepdim=True)
    sat_scale = _u(0.7, 1.3)
    images = gray + (images - gray) * sat_scale
    # Random grayscale (10%)
    gs_mask = (torch.rand(B, 1, 1, 1, device=device, dtype=dtype) < 0.10).to(dtype)
    images = images * (1.0 - gs_mask) + images.mean(dim=1, keepdim=True) * gs_mask
    # Gaussian noise
    sigma = _u(0.0, 0.03)
    images = images + torch.randn_like(images) * sigma

    return images.clamp_(0.0, 1.0)


def split_indices(count: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(count * val_fraction))) if count > 1 and val_fraction > 0 else 0
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        train_indices = indices
        val_indices = []
    return train_indices, val_indices


def resolve_valid_homography_indices(dataset: GTAUAVDataset, homography_dir: str | Path) -> List[int]:
    homography_dir = Path(homography_dir)
    cache_path = homography_dir if homography_dir.suffix == ".npz" else homography_dir / "homographies.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Homography cache not found: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)
    valid = np.asarray(data["valid"], dtype=bool)
    if "drone_paths" in data:
        drone_paths = np.asarray(data["drone_paths"], dtype=object)
    else:
        drone_paths = np.array([], dtype=object)

    if len(valid) == len(dataset.samples):
        indices = [index for index, is_valid in enumerate(valid.tolist()) if is_valid]
        if indices:
            return indices

    valid_by_path: dict[str, bool] = {}
    for path_value, is_valid in zip(drone_paths.tolist(), valid.tolist()):
        valid_by_path[normalize_path_key(path_value)] = bool(is_valid)

    indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if valid_by_path.get(normalize_path_key(sample.drone_img_path), False)
    ]
    if not indices:
        raise ValueError(
            f"No valid homography-aligned samples found in dataset for cache: {cache_path}. "
            "Check that --split and --dataset-root match the homography precompute inputs."
        )
    return indices


def load_homography_targets(homography_dir: str | Path, model_size: Tuple[int, int] = (224, 224)) -> dict[str, np.ndarray]:
    """Load homographies and normalize them to the model input `model_size`.

    Returns a dict mapping normalized drone image path keys to 3x3 homography matrices
    that map model-space drone pixel coordinates to model-space satellite pixel coordinates.
    """
    homography_dir = Path(homography_dir)
    cache_path = homography_dir if homography_dir.suffix == ".npz" else homography_dir / "homographies.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Homography cache not found: {cache_path}")

    data = np.load(cache_path, allow_pickle=True)
    homographies = np.asarray(data["homographies"], dtype=np.float32)
    valid = np.asarray(data["valid"], dtype=bool)
    drone_paths = np.asarray(data.get("drone_paths", []), dtype=object)
    sat_paths = np.asarray(data.get("sat_paths", []), dtype=object)

    model_w, model_h = int(model_size[0]), int(model_size[1])
    targets: dict[str, np.ndarray] = {}
    for path_value, sat_value, matrix, is_valid in zip(drone_paths.tolist(), sat_paths.tolist(), homographies, valid.tolist()):
        if not is_valid:
            continue
        drone_p = Path(str(path_value))
        sat_p = Path(str(sat_value)) if sat_value else None
        # Read original image sizes; fall back to model size if unavailable
        try:
            d_img = cv2.imread(str(drone_p))
            if d_img is None:
                raise FileNotFoundError
            h0, w0 = d_img.shape[:2]
        except Exception:
            w0, h0 = model_w, model_h

        if sat_p is not None and sat_p.exists():
            try:
                s_img = cv2.imread(str(sat_p))
                if s_img is None:
                    raise FileNotFoundError
                hs0, ws0 = s_img.shape[:2]
            except Exception:
                ws0, hs0 = model_w, model_h
        else:
            ws0, hs0 = model_w, model_h

        # Scaling matrices: convert model coords -> original pixels and vice versa
        A_d = np.array([[w0 / model_w, 0.0, 0.0], [0.0, h0 / model_h, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        A_s = np.array([[ws0 / model_w, 0.0, 0.0], [0.0, hs0 / model_h, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        # H maps drone_orig -> sat_orig. Convert to model-space: H' = A_s^{-1} * H * A_d
        try:
            A_s_inv = np.linalg.inv(A_s)
            H_mod = A_s_inv @ matrix @ A_d
        except Exception:
            # Fallback: use raw matrix
            H_mod = matrix

        targets[normalize_path_key(path_value)] = H_mod.astype(np.float32)

    if not targets:
        raise ValueError(f"No valid homography targets found in {cache_path}")
    return targets


def load_sat_path_map(homography_dir: str | Path) -> dict[str, str]:
    """Return a dict mapping normalized drone path key → satellite path string."""
    homography_dir = Path(homography_dir)
    cache_path = homography_dir if homography_dir.suffix == ".npz" else homography_dir / "homographies.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Homography cache not found: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    drone_paths = np.asarray(data.get("drone_paths", []), dtype=object)
    sat_paths = np.asarray(data.get("sat_paths", []), dtype=object)
    valid = np.asarray(data["valid"], dtype=bool)
    result: dict[str, str] = {}
    for dp, sp, is_valid in zip(drone_paths.tolist(), sat_paths.tolist(), valid.tolist()):
        if is_valid and sp:
            result[normalize_path_key(dp)] = str(sp)
    return result


def _load_image_as_tensor(path: str, size: int) -> torch.Tensor | None:
    """Load an image file and return a (1, 3, size, size) float32 tensor, or None on failure."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def normalize_path_key(value: str | Path) -> str:
    return Path(str(value)).as_posix().lower()


def make_loader(dataset, batch_size: int, num_workers: int, pin_memory: bool, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
        collate_fn=collate_batch,
    )


def collate_batch(batch: Sequence[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch])
    metas = [item["meta"] for item in batch]
    return {"image": images, "meta": metas}


def collate_window_batch(batch: Sequence[dict]) -> dict:
    images = torch.stack([item["images"] for item in batch])
    metas = [item["meta"] for item in batch]
    return {"images": images, "meta": metas}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
