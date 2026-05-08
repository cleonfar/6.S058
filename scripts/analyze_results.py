"""Evaluate a checkpoint and break down error by confidence tier and image index.

Usage:
    python scripts/analyze_results.py \
        --checkpoint runs/baseline_v6/best.pt \
        --split University-Release/university_test.json \
        --dataset-root University-Release \
        --homography-dir data/homography_university_test \
        --use-gpu

Produces:
    runs/baseline_v6/analysis_test/
        summary.txt          - overall numbers
        by_confidence.png    - bar chart: overhead vs oblique
        by_image_index.png   - line chart: error vs camera orbit angle
        qualitative/         - 6 sample overlays (3 best, 3 worst)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from gtauav_loc.train_baseline import (
    SpatialEncoder,
    CorrelationVolume,
    collate_batch,
    resolve_valid_homography_indices,
    load_homography_targets,
    load_sat_path_map,
    normalize_path_key,
    _load_image_as_tensor,
    normalize_imagenet,
)
from gtauav_loc.homography import (
    SpatialHomographyHead,
    apply_homography_to_points,
    compute_homography_reprojection_loss,
)
from gtauav_loc.torch_dataset import GTAUAVDataset
from gtauav_loc.utils.device import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="University-Release/university_test.json")
    p.add_argument("--dataset-root", default="University-Release")
    p.add_argument("--homography-dir", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--warp-size", type=int, default=224)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def corners_to_quad(corners_norm: np.ndarray, img_size: int) -> np.ndarray:
    """(8,) normalised corners → (4,2) pixel coords."""
    pts = corners_norm.reshape(4, 2) * img_size
    return pts.astype(np.float32)


def draw_quad(img: np.ndarray, pts: np.ndarray, color, thickness: int = 2) -> np.ndarray:
    """Draw a closed quadrilateral on img (in-place copy)."""
    img = img.copy()
    order = [0, 1, 2, 3, 0]
    for a, b in zip(order, order[1:]):
        p1 = tuple(map(int, pts[a]))
        p2 = tuple(map(int, pts[b]))
        cv2.line(img, p1, p2, color, thickness)
    return img


def _blend_warp(drone_img: np.ndarray, sat_img: np.ndarray,
                H: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Warp drone_img by H into satellite space and alpha-blend over sat_img."""
    W = sat_img.shape[1]
    H_f = H.astype(np.float64)
    warped = cv2.warpPerspective(drone_img, H_f, (W, W))
    mask   = cv2.warpPerspective(
        np.ones((W, W), dtype=np.uint8) * 255, H_f, (W, W)
    )
    out = sat_img.copy()
    m = mask > 0
    out[m] = (alpha * warped[m].astype(float) +
              (1.0 - alpha) * sat_img[m].astype(float)).astype(np.uint8)
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_qualitative(
    drone_path: str,
    sat_path: str,
    pred_corners: np.ndarray,
    target_h: np.ndarray,
    error_px: float,
    out_path: Path,
    warp_size: int,
) -> None:
    drone_img = cv2.imread(drone_path)
    sat_img   = cv2.imread(sat_path)
    if drone_img is None or sat_img is None:
        return

    W = warp_size
    drone_img = cv2.resize(drone_img, (W, W))
    sat_img   = cv2.resize(sat_img,   (W, W))

    # Source corners (TL, TR, BR, BL)
    src_corners = np.array([[0, 0], [W, 0], [W, W], [0, W]], dtype=np.float32)

    # GT corners via target homography
    H_t   = torch.tensor(target_h, dtype=torch.float32).unsqueeze(0)
    src_t = torch.tensor(src_corners, dtype=torch.float32).unsqueeze(0)
    tgt_pts = apply_homography_to_points(H_t, src_t).squeeze(0).numpy()  # (4,2)

    # Predicted corners
    pred_pts = corners_to_quad(pred_corners, W)  # (4,2)

    # --- panel A: original drone ---
    panel_a = _label(drone_img, "Drone (query)")

    # --- panel B: satellite with GT quad (green) and predicted quad (red) ---
    panel_b = draw_quad(sat_img, tgt_pts,  (30, 200, 30),  2)
    panel_b = draw_quad(panel_b, pred_pts, (30,  30, 220), 2)
    panel_b = _label(panel_b, "Satellite  |  GT=green  Pred=red")

    # --- panel C: predicted composite (drone warped by predicted H) ---
    H_pred  = cv2.getPerspectiveTransform(src_corners, pred_pts)
    panel_c = _blend_warp(drone_img, sat_img, H_pred)
    panel_c = _label(panel_c, "Predicted composite")

    # --- panel D: GT composite (drone warped by GT H) ---
    panel_d = _blend_warp(drone_img, sat_img, np.array(target_h, dtype=np.float64))
    panel_d = _label(panel_d, "GT composite")

    # 2×2 grid + error banner at bottom
    top = np.hstack([panel_a, panel_b])
    bot = np.hstack([panel_c, panel_d])
    grid = np.vstack([top, bot])

    # Error banner
    banner = np.zeros((28, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, f"corner error = {error_px:.1f} px  |  {Path(drone_path).parent.name}/{Path(drone_path).name}",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    panel = np.vstack([grid, banner])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = get_device(prefer_gpu=args.use_gpu)
    out_dir = Path(args.checkpoint).parent / f"analysis_{Path(args.split).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load model ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    enc  = SpatialEncoder().to(device)
    enc.load_state_dict(ckpt["model_state"])
    enc.eval()
    corr = CorrelationVolume().to(device)
    _map = args.warp_size // 16
    head = SpatialHomographyHead(in_channels=_map * _map).to(device)
    head.load_state_dict(ckpt["homography_head_state"])
    head.eval()

    # ---- dataset ----
    ds = GTAUAVDataset(
        split_json=args.split,
        dataset_root=args.dataset_root,
        mode="single",
        transform=T.Compose([T.Resize((args.warp_size, args.warp_size)), T.ToTensor()]),
        warp_overhead=False,
    )
    hom_dir = args.homography_dir
    valid_indices = resolve_valid_homography_indices(ds, hom_dir)
    homography_targets = load_homography_targets(hom_dir, model_size=(args.warp_size, args.warp_size))
    sat_path_map       = load_sat_path_map(hom_dir)

    # Load confidence from npz for stratification
    npz = np.load(Path(hom_dir) / "homographies.npz", allow_pickle=True)
    drone_paths_all = np.asarray(npz["drone_paths"], dtype=object)
    confs_all       = np.asarray(npz["confidence"],  dtype=float) if "confidence" in npz else None
    conf_by_key: dict[str, float] = {}
    if confs_all is not None:
        for p, c in zip(drone_paths_all.tolist(), confs_all.tolist()):
            conf_by_key[normalize_path_key(p)] = c

    eval_ds = Subset(ds, valid_indices)
    loader  = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate_batch)

    # ---- inference ----
    records = []  # list of dicts with key, error_px, confidence, image_idx, drone_path, sat_path, pred_corners

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            images = batch["image"].to(device)
            metas  = batch["meta"]
            valid_idx, target_h_list, sat_tensors, keys, drone_ps, sat_ps = [], [], [], [], [], []
            for i, meta in enumerate(metas):
                key = normalize_path_key(meta.get("drone_img_path", ""))
                th  = homography_targets.get(key)
                sp  = sat_path_map.get(key)
                if th is None or sp is None:
                    continue
                st = _load_image_as_tensor(sp, args.warp_size)
                if st is None:
                    continue
                valid_idx.append(i)
                target_h_list.append(th)
                sat_tensors.append(st)
                keys.append(key)
                drone_ps.append(meta.get("drone_img_path", ""))
                sat_ps.append(sp)

            if not valid_idx:
                continue

            sat_imgs = torch.cat(sat_tensors, 0).to(device)
            idx_t    = torch.tensor(valid_idx, dtype=torch.long, device=device)

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                dm   = enc(normalize_imagenet(torch.index_select(images, 0, idx_t)))
                sm   = enc(normalize_imagenet(sat_imgs))
                pred = head(corr(dm, sm))  # (N, 8)

            tgt = torch.from_numpy(np.stack(target_h_list)).to(device=device, dtype=pred.dtype)

            # Per-sample error in pixels
            W = float(args.warp_size)
            src_px = torch.tensor([[0,0],[W,0],[W,W],[0,W]], dtype=pred.dtype, device=device)
            src_px = src_px.unsqueeze(0).expand(len(valid_idx), -1, -1)
            tgt_corners_px = apply_homography_to_points(tgt, src_px)  # (N,4,2)
            pred_corners_px = pred.view(-1, 4, 2) * W
            per_sample_err = (pred_corners_px - tgt_corners_px).norm(dim=-1).mean(dim=-1)  # (N,)

            for j, (key, dp, sp) in enumerate(zip(keys, drone_ps, sat_ps)):
                err_px = float(per_sample_err[j].cpu())
                conf   = conf_by_key.get(key, np.nan)
                # Parse image index from filename: "1318/image-07.jpeg" -> 7
                img_name = Path(dp).name  # e.g. "image-07.jpeg"
                try:
                    img_idx = int(img_name.split("-")[1].split(".")[0])
                except Exception:
                    img_idx = -1
                records.append({
                    "key":          key,
                    "error_px":     err_px,
                    "confidence":   conf,
                    "image_idx":    img_idx,
                    "drone_path":   dp,
                    "sat_path":     sp,
                    "pred_corners": pred[j].cpu().numpy(),
                    "target_h":     target_h_list[j],
                })

    errors   = np.array([r["error_px"]   for r in records])
    confs    = np.array([r["confidence"] for r in records])
    img_idxs = np.array([r["image_idx"]  for r in records])

    mean_err = errors.mean()
    print(f"\nOverall: n={len(errors)} mean_err={mean_err:.2f}px median={np.median(errors):.2f}px")

    # ---- 1. By confidence tier ----
    strict_mask  = confs >= 0.99   # confidence == 1.0 (overhead SIFT)
    oblique_mask = confs < 0.99    # confidence == 0.5 (oblique-recovered)
    strict_err   = errors[strict_mask]
    oblique_err  = errors[oblique_mask]

    print(f"\nBy SIFT confidence:")
    print(f"  Overhead (conf=1.0): n={strict_mask.sum():5d}  mean={strict_err.mean():.2f}px  median={np.median(strict_err):.2f}px")
    print(f"  Oblique  (conf=0.5): n={oblique_mask.sum():5d}  mean={oblique_err.mean():.2f}px  median={np.median(oblique_err):.2f}px")

    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ["Overhead\n(SIFT conf=1.0)", "Oblique\n(SIFT conf=0.5)"]
    means  = [strict_err.mean() if len(strict_err) else 0,
              oblique_err.mean() if len(oblique_err) else 0]
    medians = [np.median(strict_err) if len(strict_err) else 0,
               np.median(oblique_err) if len(oblique_err) else 0]
    x = np.arange(len(categories))
    bars = ax.bar(x - 0.2, means,   0.35, label="Mean",   color="#4C72B0")
    bars2= ax.bar(x + 0.2, medians, 0.35, label="Median", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("Corner reprojection error (px)")
    ax.set_title("Prediction error by SIFT confidence tier")
    ax.legend()
    ax.bar_label(bars,  fmt="%.1f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "by_confidence.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'by_confidence.png'}")

    # ---- 2. By image index (camera orbit angle) ----
    unique_idxs = sorted(i for i in np.unique(img_idxs) if i > 0)
    idx_means   = []
    idx_counts  = []
    for ii in unique_idxs:
        mask = img_idxs == ii
        idx_means.append(errors[mask].mean())
        idx_counts.append(mask.sum())

    print(f"\nBy image index (orbit position):")
    for ii, m, n in zip(unique_idxs, idx_means, idx_counts):
        print(f"  image-{ii:02d}: n={n:5d}  mean={m:.2f}px")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(unique_idxs, idx_means, marker="o", color="#4C72B0", linewidth=1.5)
    ax.set_xlabel("Image index (camera orbit position)")
    ax.set_ylabel("Mean corner reprojection error (px)")
    ax.set_title("Prediction error by drone orbit angle")
    ax.set_xticks(unique_idxs)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "by_image_index.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'by_image_index.png'}")

    # ---- 3. Qualitative: 3 best + 3 worst + 1 near each percentile ----
    sorted_records = sorted(records, key=lambda r: r["error_px"])
    qual_dir = out_dir / "qualitative"
    qual_dir.mkdir(exist_ok=True)
    for rank, rec in enumerate(sorted_records[:3]):
        save_qualitative(
            rec["drone_path"], rec["sat_path"],
            rec["pred_corners"], rec["target_h"],
            rec["error_px"],
            qual_dir / f"best_{rank+1}_err{rec['error_px']:.1f}px.jpg",
            args.warp_size,
        )
    for rank, rec in enumerate(sorted_records[-3:]):
        save_qualitative(
            rec["drone_path"], rec["sat_path"],
            rec["pred_corners"], rec["target_h"],
            rec["error_px"],
            qual_dir / f"worst_{rank+1}_err{rec['error_px']:.1f}px.jpg",
            args.warp_size,
        )
    # Percentile samples: pick the record whose error is closest to p25/p50/p75
    for pct in [25, 50, 75]:
        target_err = float(np.percentile(errors, pct))
        rec = min(sorted_records, key=lambda r: abs(r["error_px"] - target_err))
        save_qualitative(
            rec["drone_path"], rec["sat_path"],
            rec["pred_corners"], rec["target_h"],
            rec["error_px"],
            qual_dir / f"p{pct}_err{rec['error_px']:.1f}px.jpg",
            args.warp_size,
        )
    print(f"Saved qualitative examples to {qual_dir}/")

    # ---- 4. Summary text ----
    summary = (
        f"Checkpoint : {args.checkpoint}\n"
        f"Split      : {args.split}\n"
        f"N evaluated: {len(errors)}\n"
        f"\nOverall\n"
        f"  mean   : {mean_err:.2f}px\n"
        f"  median : {np.median(errors):.2f}px\n"
        f"  p25/p75: {np.percentile(errors,25):.2f}px / {np.percentile(errors,75):.2f}px\n"
        f"\nBy SIFT tier\n"
        f"  Overhead (n={strict_mask.sum()}) : mean={strict_err.mean():.2f}px  median={np.median(strict_err):.2f}px\n"
        f"  Oblique  (n={oblique_mask.sum()}): mean={oblique_err.mean():.2f}px  median={np.median(oblique_err):.2f}px\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    print(f"\nSaved {out_dir / 'summary.txt'}")
    print(summary)


if __name__ == "__main__":
    main()
