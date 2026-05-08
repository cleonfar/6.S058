from pathlib import Path

from gtauav_loc import group_into_sequences, load_split, make_sequence_batches


def test_group_into_sequences_uses_filename_prefix() -> None:
    dataset_root = Path(__file__).resolve().parents[1] / "dataset"
    split_path = dataset_root / "same-area-drone2sate-train.json"

    samples = load_split(split_path, dataset_root)
    sequences = group_into_sequences(samples)

    assert len(samples) > 0
    assert len(sequences) == 6
    assert sequences[0][0].drone_img_path.name.startswith("100_")


def test_make_sequence_batches_creates_windows() -> None:
    dataset_root = Path(__file__).resolve().parents[1] / "dataset"
    split_path = dataset_root / "same-area-drone2sate-train.json"

    samples = load_split(split_path, dataset_root)
    batches = make_sequence_batches(samples, window_size=3, stride=2)

    assert batches
    assert all(len(batch) == 3 for batch in batches)
