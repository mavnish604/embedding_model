import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_image_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


class ImageTextPairDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        *,
        image_root=None,
        image_transform=None,
        text_key_candidates=("text", "caption"),
        image_key_candidates=("image", "image_path"),
    ):
        self.metadata_path = Path(metadata_path)
        self.image_root = Path(image_root) if image_root is not None else self.metadata_path.parent
        self.image_transform = image_transform
        self.text_key_candidates = text_key_candidates
        self.image_key_candidates = image_key_candidates
        self.records = self._load_records()

    def _load_records(self):
        records = []
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {self.metadata_path}"
                    ) from exc

                image_key = next(
                    (key for key in self.image_key_candidates if key in record),
                    None,
                )
                text_key = next(
                    (key for key in self.text_key_candidates if key in record),
                    None,
                )
                if image_key is None or text_key is None:
                    raise ValueError(
                        f"Each metadata row must include one of {self.image_key_candidates} "
                        f"and one of {self.text_key_candidates}. Failed on line {line_number}."
                    )

                records.append(
                    {
                        "image_path": record[image_key],
                        "text": record[text_key],
                    }
                )

        if not records:
            raise ValueError(f"No records found in {self.metadata_path}")

        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "pixel_values": image,
            "text": record["text"],
            "image_path": str(image_path),
        }


class ImageTextCollator:
    def __init__(self, tokenizer, *, max_length=96):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
        texts = [item["text"] for item in batch]

        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "texts": texts,
        }
