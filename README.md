# UAV Sequential Localization (University-1652)

This repository is a lightweight starting point for UAV-to-satellite image localization with the [University-1652](https://github.com/layumi/University1652-Baseline) dataset.

> **Note on naming:** Many files, modules, and CLI entry points are prefixed with `gtauav` or similar. This is because the project was originally built around the GTA-UAV dataset before switching to University-1652. The filenames were not updated to avoid breaking existing scripts and runs.

## What is included

- Dataset loading helpers for the University-1652 JSON annotation files
- Sequence reconstruction from drone image names
- Sliding-window batch generation for sequential training
- A small CLI to inspect the reconstructed sequences
- A basic test to validate the sequence grouping logic

## Layout

- `dataset/` contains the University-1652 images and split JSON files
- `src/gtauav_loc/` contains the importable Python package
- `tests/` contains a small unit test

## Setup

Create and activate a Python environment, then install the package in editable mode:

```bash
pip install -e .
pip install -e .[dev]
```

## Quick start

Inspect the reconstructed sequence layout:

```bash
gtauav-loc summary --split dataset/same-area-drone2sate-train.json
```

Use the library from Python:

```python
from gtauav_loc import load_split, group_into_sequences

samples = load_split("dataset/same-area-drone2sate-train.json", "dataset")
sequences = group_into_sequences(samples)
print(len(samples), len(sequences))
```

Train the first retrieval baseline with cached satellite embeddings and GPU warp preprocessing:

```powershell
python -m gtauav_loc.train_baseline --split dataset/same-area-drone2sate-train.json --dataset-root dataset --precompute-dir data/precompute_res18_sift --out-dir runs/baseline_res18 --use-gpu --warp-backend torch --epochs 10 --batch-size 16
```

The trainer uses mixed precision on CUDA automatically, writes checkpoints to `runs/baseline`, and reports per-epoch progress plus loss/ETA information.

Evaluate a trained checkpoint with the shared benchmark harness:

```powershell
gtauav-eval --checkpoint runs/baseline/best.pt --split dataset/same-area-drone2sate-test.json --dataset-root dataset --precompute-dir data/precompute --use-gpu --warp-backend torch
```

If you enable `--rerank-sift-ransac`, make sure the precompute directory was built without `--skip-sift`, because the reranker needs the saved satellite keypoints and descriptors.

This uses the same loading and preprocessing path for every model variant, so baseline and sequential runs can be compared under the same conditions.

Satellite precompute now uses the learned GPU DISK backend by default for local descriptors. It still saves the same precompute artifacts, but there is no backend selection flag anymore.

```powershell
python -m gtauav_loc.satellite_precompute --dataset-root dataset --out-dir data/precompute_res18_sift --batch-size 64 --use-gpu
```

For the learned descriptor backend, make sure `torch` and `kornia` are installed in the venv.

## Sequential Training

Train the sequential models (ordered by segment and frame id) with and without the confidence head:

```powershell
gtauav-train-sequential --split dataset/same-area-drone2sate-train.json --dataset-root dataset --precompute-dir data/precompute_res18_sift --out-dir runs/sequential_noconf --use-gpu --warp-backend torch --epochs 10 --batch-size 8 --window-size 4 --stride 1 --use-location-feedback
gtauav-train-sequential --split dataset/same-area-drone2sate-train.json --dataset-root dataset --precompute-dir data/precompute_res18_sift --out-dir runs/sequential_conf --use-gpu --warp-backend torch --epochs 10 --batch-size 8 --window-size 4 --stride 1 --use-location-feedback --with-confidence
```

### Unified Loss Across All Models

All models (baseline and sequential variants) use the same objective: **retrieval_loss + coherence_loss_weight × coherence_penalty**

- **Baseline**: Coherence loss is zero (single-frame predictions have no temporal dimension), so it effectively uses retrieval loss only.
- **Sequential (no confidence)**: Coherence loss penalizes large spatial jumps between consecutive predicted tiles.
- **Sequential (with confidence)**: Coherence loss + confidence loss, where confidence reflects match quality.

This unified loss structure now keeps training focused on contrastive tile matching. The sequential model can still use previous location estimates as recurrent input and can optionally train a confidence head, but there is no separate location or coherence regression loss anymore.

For the sequential models, the training script groups samples by image-name prefix, sorts each sequence by the trailing frame number, and then builds sliding windows. Validation splits are done at the sequence level so no frames from the same trajectory leak between train and val.

## Next step

The current scaffold is ready for adding a sequential localization model that consumes prior predictions as context during inference.
