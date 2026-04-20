import argparse
import json
from pathlib import Path

from torchvision.datasets import CIFAR10
from tqdm import tqdm


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "clip" / "cifar10_train"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download CIFAR-10 and export it as image-text pairs for CLIP-style training."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=10000)
    return parser.parse_args()


def build_caption(class_name):
    article = "an" if class_name[0].lower() in "aeiou" else "a"
    return f"{article} photo of {article} {class_name}"


def main():
    args = parse_args()
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = CIFAR10(root=str(raw_dir), train=True, download=True)
    total = min(args.max_samples, len(dataset))
    metadata_path = output_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as handle:
        for index in tqdm(range(total), desc="Exporting CIFAR-10"):
            image, label = dataset[index]
            class_name = dataset.classes[label]
            file_name = f"{index:05d}_{class_name}.png"
            image_path = images_dir / file_name
            image.save(image_path)

            record = {
                "image": str(image_path.relative_to(output_dir)).replace("\\", "/"),
                "text": build_caption(class_name),
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Saved dataset to: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Images: {images_dir}")
    print(f"Exported rows: {total}")


if __name__ == "__main__":
    main()
