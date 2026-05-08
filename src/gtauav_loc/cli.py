from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_split
from .sequences import group_into_sequences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gtauav-loc", description="Inspect GTA-UAV sequential localization data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Show sample and sequence counts")
    summary.add_argument("--split", required=True, help="Path to a GTA-UAV split JSON file")
    summary.add_argument("--dataset-root", default="dataset", help="Root directory containing the dataset assets")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "summary":
        samples = load_split(Path(args.split), Path(args.dataset_root))
        sequences = group_into_sequences(samples)
        print(f"samples={len(samples)} sequences={len(sequences)}")
        if sequences:
            preview = sequences[0][:3]
            print("preview:")
            for sample in preview:
                print(f"  {sample.drone_img_path.name} -> {sample.drone_loc_x_y}")


if __name__ == "__main__":
    main()
