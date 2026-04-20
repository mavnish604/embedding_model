import argparse
import io
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


DEFAULT_DATASET = "intro/flickr8k"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "clip" / "flickr8k_train"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and export an image-text dataset to metadata.jsonl format."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def resolve_text(row):
    if isinstance(row.get("text"), str) and row["text"].strip():
        return row["text"].strip()

    if isinstance(row.get("caption"), str) and row["caption"].strip():
        return row["caption"].strip()

    if isinstance(row.get("query"), str) and row["query"].strip():
        return row["query"].strip()

    captions = row.get("captions")
    if isinstance(captions, list):
        for caption in captions:
            if isinstance(caption, str) and caption.strip():
                return caption.strip()

    caption_keys = sorted(key for key in row if key.startswith("caption_"))
    for key in caption_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError("Could not find a usable text caption in dataset row.")


def resolve_image(row):
    image = row.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path") is not None:
            return Image.open(image["path"]).convert("RGB")

    raise ValueError("Could not find a usable image in dataset row.")


def resolve_image_name(row, index):
    file_name = row.get("file_name")
    if isinstance(file_name, str) and file_name.strip():
        original = Path(file_name.strip()).name
        suffix = Path(original).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            return original
        return f"{Path(original).stem}.jpg"

    return f"image_{index:06d}.jpg"


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=95)
        return

    if suffix == ".png":
        image.save(path, format="PNG")
        return

    if suffix == ".webp":
        image.save(path, format="WEBP", quality=95)
        return

    image.save(path.with_suffix(".jpg"), format="JPEG", quality=95)


def main():
    args = parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    output_dir = args.output_dir
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.jsonl"
    exported_count = 0

    with metadata_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(tqdm(dataset, desc="Exporting image-text data")):
            text = resolve_text(row)
            image = resolve_image(row)
            image_name = resolve_image_name(row, index)
            image_path = images_dir / image_name
            save_image(image, image_path)

            relative_path = image_path.relative_to(output_dir)
            handle.write(
                '{"image": "%s", "text": %s}\n'
                % (
                    str(relative_path).replace("\\", "/"),
                    json.dumps(text, ensure_ascii=True),
                )
            )
            exported_count += 1

    print(f"Saved dataset to: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Images: {images_dir}")
    print(f"Exported rows: {exported_count}")


if __name__ == "__main__":
    main()
