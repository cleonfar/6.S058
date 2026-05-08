from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class SatelliteEmbeddingStore:
    root: Path
    files: Tuple[str, ...]
    embeddings: np.ndarray
    _index_by_stem: Dict[str, int]
    _tile_xy: Dict[str, Tuple[float, float]]

    @classmethod
    def load(cls, root: str | Path) -> "SatelliteEmbeddingStore":
        root_path = Path(root)
        file_list_path = root_path / "files.npy"
        if file_list_path.exists():
            files = tuple(str(value) for value in np.load(file_list_path, allow_pickle=True).tolist())
            npz_paths = [root_path / (Path(path).stem + ".npz") for path in files]
        else:
            npz_paths = sorted(root_path.glob("*.npz"))
            files = tuple(str(path.with_suffix(".png")) for path in npz_paths)

        embeddings: List[np.ndarray] = []
        index_by_stem: Dict[str, int] = {}
        tile_xy: Dict[str, Tuple[float, float]] = {}
        for idx, npz_path in enumerate(npz_paths):
            data = np.load(npz_path)
            embedding = np.asarray(data["emb"], dtype=np.float32).reshape(-1)
            embeddings.append(embedding)
            stem = npz_path.stem
            index_by_stem[stem] = idx
            tile_xy[stem] = _parse_tile_coordinates(stem)

        if not embeddings:
            raise FileNotFoundError(f"No cached satellite embeddings found in {root_path}")

        return cls(
            root=root_path,
            files=files,
            embeddings=np.stack(embeddings, axis=0),
            _index_by_stem=index_by_stem,
            _tile_xy=tile_xy,
        )

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def has(self, satellite_name: str) -> bool:
        return Path(satellite_name).stem in self._index_by_stem

    def get(self, satellite_name: str) -> np.ndarray:
        stem = Path(satellite_name).stem
        if stem not in self._index_by_stem:
            raise KeyError(f"Missing cached embedding for {satellite_name}")
        return self.embeddings[self._index_by_stem[stem]]

    def batch_get(self, satellite_names: Sequence[str]) -> np.ndarray:
        vectors = [self.get(name) for name in satellite_names]
        return np.stack(vectors, axis=0)

    def resolve_target(self, positive_names: Sequence[str], positive_weights: Sequence[float]) -> np.ndarray:
        candidates = [name for name in positive_names if self.has(name)]
        if not candidates:
            raise KeyError("None of the requested satellite names exist in the precompute cache")

        if positive_weights and len(positive_weights) == len(positive_names):
            weights = np.asarray([weight for name, weight in zip(positive_names, positive_weights) if self.has(name)], dtype=np.float32)
        else:
            weights = np.ones(len(candidates), dtype=np.float32)

        embeddings = self.batch_get(candidates)
        weights = weights / max(weights.sum(), 1e-6)
        target = (embeddings * weights[:, None]).sum(axis=0)
        norm = np.linalg.norm(target)
        return target / max(norm, 1e-6)

    def get_tile_location(self, satellite_name: str) -> Tuple[float, float]:
        """Get the (x, y) center coordinates of a satellite tile."""
        stem = Path(satellite_name).stem
        if stem not in self._tile_xy:
            raise KeyError(f"Missing tile coordinates for {satellite_name}")
        return self._tile_xy[stem]

    def batch_get_tile_locations(self, satellite_names: Sequence[str]) -> np.ndarray:
        """Get tile locations for multiple satellites as an Nx2 array."""
        locations = np.array([self.get_tile_location(name) for name in satellite_names], dtype=np.float32)
        return locations


def _parse_tile_coordinates(tile_stem: str) -> Tuple[float, float]:
    """Parse real world (x, y) coordinates from a satellite tile filename.

    The satellite tiles are named as Z_W_X_Y. X/Y are grid coordinates, but the
    evaluation and temporal losses need the corresponding world-space center.
    The dataset uses a fixed zoom-level grid where each level has a known scale
    and offset.
    
    For datasets without real coordinates (e.g., University-1652), returns (0.0, 0.0).
    """
    parts = tile_stem.split("_")
    if len(parts) < 4:
        # Not in Z_W_X_Y format; assume no real coordinates (e.g., University-1652)
        return (0.0, 0.0)

    try:
        z_level = parts[0]
        grid_x = float(parts[2])
        grid_y = float(parts[3])

        z_level_params = {
            "4": (691.2, 345.6),
            "5": (345.6, 172.8),
            "6": (172.8, 86.4),
            "7": (86.4, 43.2),
        }
        if z_level not in z_level_params:
            # Unknown format; return dummy coordinates
            return (0.0, 0.0)

        scale, offset = z_level_params[z_level]
        return scale * grid_x + offset, scale * grid_y + offset
    except (ValueError, IndexError):
        # Parse error; return dummy coordinates
        return (0.0, 0.0)
