#!/usr/bin/env python
"""Analyze satellite tile embedding diversity to detect near-duplicate tiles."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from gtauav_loc.satellite_store import SatelliteEmbeddingStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tile embedding diversity")
    parser.add_argument("--precompute-dir", default="data/precompute")
    parser.add_argument("--percentile", type=float, default=90.0, help="Percentile threshold for 'similar' tile pairs")
    args = parser.parse_args()

    store = SatelliteEmbeddingStore.load(args.precompute_dir)
    embeddings = store.embeddings
    stems = [Path(p).stem for p in store.files]

    print(f"Loaded {len(embeddings)} tiles")

    # Compute pairwise cosine distances
    print("Computing pairwise distances...")
    distances = cosine_distances(embeddings)
    np.fill_diagonal(distances, np.inf)  # Ignore self-distance

    # Get distance distribution
    min_dist = np.min(distances[np.isfinite(distances)])
    max_dist = np.max(distances)
    median_dist = np.median(distances[np.isfinite(distances)])
    percentile_dist = np.percentile(distances[np.isfinite(distances)], args.percentile)

    print(f"\nDistance statistics:")
    print(f"  Min distance (closest pair): {min_dist:.6f}")
    print(f"  Median distance: {median_dist:.6f}")
    print(f"  {args.percentile}th percentile: {percentile_dist:.6f}")
    print(f"  Max distance: {max_dist:.6f}")

    # Count highly similar pairs
    similar_threshold = percentile_dist
    similar_pairs = np.sum(distances < similar_threshold) // 2  # Divide by 2 because matrix is symmetric
    total_pairs = len(embeddings) * (len(embeddings) - 1) // 2
    similar_fraction = similar_pairs / total_pairs if total_pairs > 0 else 0

    print(f"\nSimilar tile pairs (distance < {similar_threshold:.6f}): {similar_pairs} / {total_pairs} ({similar_fraction * 100:.2f}%)")

    # Find most similar tiles
    print(f"\nTop 10 most similar tile pairs:")
    flat_indices = np.argsort(distances, axis=None)[:10]
    for idx in flat_indices:
        i, j = np.unravel_index(idx, distances.shape)
        if i < j:
            print(f"  {stems[i]} <-> {stems[j]}: distance={distances[i, j]:.6f}")


if __name__ == "__main__":
    main()
