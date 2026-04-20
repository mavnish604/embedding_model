import argparse
from pathlib import Path

from dataset import HF_CACHE_DIR
from dataset import PROCESSED_DATA_DIR
from dataset import prepare_nli_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and preprocess the SNLI + MultiNLI datasets."
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=HF_CACHE_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = prepare_nli_dataset(output_dir=args.output_dir, cache_dir=args.cache_dir)

    print(f"Saved processed dataset to: {args.output_dir}")
    for split_name, split_dataset in dataset.items():
        print(f"{split_name}: {len(split_dataset)} rows")


if __name__ == "__main__":
    main()
