from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence

from .data import LocalizationSample


def group_into_sequences(samples: Sequence[LocalizationSample]) -> List[List[LocalizationSample]]:
    grouped = {}
    for sample in samples:
        try:
            segment_id, frame_id = _parse_drone_name(sample.drone_img_path.name)
        except ValueError:
            # University-1652 style folders store images under per-location directories
            # with filenames such as image-01.jpeg. In that case, treat the parent
            # folder as the sequence id and sort frames by any numeric suffix when present.
            segment_id = sample.drone_img_path.parent.name
            frame_id = _parse_fallback_frame_id(sample.drone_img_path.stem)
        grouped.setdefault(segment_id, []).append((frame_id, sample))

    sequences: List[List[LocalizationSample]] = []
    for segment_id in sorted(grouped.keys()):
        ordered = [sample for _, sample in sorted(grouped[segment_id], key=lambda item: item[0])]
        sequences.append(ordered)
    return sequences


def sliding_windows(
    sequence: Sequence[LocalizationSample],
    window_size: int,
    stride: int = 1,
    include_partial: bool = False,
) -> Iterator[List[LocalizationSample]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    sequence_length = len(sequence)
    if sequence_length == 0:
        return

    last_start = sequence_length if include_partial else sequence_length - window_size + 1
    for start in range(0, max(last_start, 0), stride):
        window = list(sequence[start : start + window_size])
        if len(window) < window_size and not include_partial:
            break
        if window:
            yield window


def make_sequence_batches(
    samples: Sequence[LocalizationSample],
    window_size: int,
    stride: int = 1,
    include_partial: bool = False,
) -> List[List[LocalizationSample]]:
    batches: List[List[LocalizationSample]] = []
    for sequence in group_into_sequences(samples):
        batches.extend(sliding_windows(sequence, window_size, stride=stride, include_partial=include_partial))
    return batches


def _parse_drone_name(drone_img_name: str) -> tuple[str, int]:
    stem = Path(drone_img_name).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected drone image name format: {drone_img_name}")
    return parts[0], int(parts[2])


def _parse_fallback_frame_id(drone_img_stem: str) -> int:
    digits = "".join(character for character in drone_img_stem if character.isdigit())
    if digits:
        return int(digits)
    return 0
