from pathlib import Path

from datasets import Dataset as HFDataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset
from datasets import load_from_disk
from torch.utils.data import Dataset


HF_CACHE_DIR = Path(__file__).resolve().parent / ".hf_cache"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent / "data" / "nli_pairs"

DATASET_LOADERS = {
    "snli": ("snli",),
    "multi_nli": ("multi_nli", "nyu-mll/multi_nli"),
    "glue_mrpc": ("glue",),
}

NLI_SOURCES = {
    "snli": {
        "splits": {
            "train": ("train",),
            "validation": ("validation",),
            "test": ("test",),
        },
        "text_key_1": "premise",
        "text_key_2": "hypothesis",
        "label_key": "label",
    },
    "multi_nli": {
        "splits": {
            "train": ("train",),
            "validation": ("validation_matched", "validation_mismatched"),
        },
        "text_key_1": "premise",
        "text_key_2": "hypothesis",
        "label_key": "label",
    },
}


def _cached_glue_arrow(split="train"):
    matches = sorted(
        Path.home().glob(
            f".cache/huggingface/datasets/glue/mrpc/*/*/glue-{split}.arrow"
        )
    )
    if matches:
        return matches[-1]

    return None


def _load_dataset_with_fallback(dataset_name, *, split=None, cache_dir=HF_CACHE_DIR):
    load_ids = DATASET_LOADERS.get(dataset_name, (dataset_name,))

    last_error = None
    for load_id in load_ids:
        try:
            if dataset_name == "glue_mrpc":
                return load_dataset("glue", "mrpc", split=split, cache_dir=str(cache_dir))

            return load_dataset(load_id, split=split, cache_dir=str(cache_dir))
        except Exception as exc:
            last_error = exc

    if last_error is None:
        raise ValueError(f"Unable to resolve dataset loader for {dataset_name}.")

    raise RuntimeError(f"Failed to load dataset {dataset_name}.") from last_error


def _normalize_label_names(dataset):
    label_feature = dataset.features.get("label")
    if label_feature is None:
        return None

    return getattr(label_feature, "names", None)


def _prepare_nli_split(dataset, *, source_name, split_name, text_key_1, text_key_2, label_key):
    label_names = _normalize_label_names(dataset)

    dataset = dataset.filter(
        lambda row: row[label_key] >= 0,
        desc=f"Filtering unlabeled rows from {source_name}:{split_name}",
    )

    def map_batch(batch):
        labels = batch[label_key]
        label_text = [
            label_names[label] if label_names is not None else str(label)
            for label in labels
        ]

        return {
            "sentence1": batch[text_key_1],
            "sentence2": batch[text_key_2],
            "label": labels,
            "label_text": label_text,
            "is_positive": [text == "entailment" for text in label_text],
            "source_dataset": [source_name] * len(labels),
            "source_split": [split_name] * len(labels),
        }

    return dataset.map(
        map_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Preprocessing {source_name}:{split_name}",
    )


def prepare_nli_dataset(output_dir=PROCESSED_DATA_DIR, cache_dir=HF_CACHE_DIR):
    cache_dir.mkdir(exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    prepared_splits = {}

    for source_name, config in NLI_SOURCES.items():
        split_groups = {}

        for split_name, source_splits in config["splits"].items():
            current_splits = []

            for source_split in source_splits:
                dataset = _load_dataset_with_fallback(
                    source_name,
                    split=source_split,
                    cache_dir=cache_dir,
                )
                current_splits.append(
                    _prepare_nli_split(
                        dataset,
                        source_name=source_name,
                        split_name=source_split,
                        text_key_1=config["text_key_1"],
                        text_key_2=config["text_key_2"],
                        label_key=config["label_key"],
                    )
                )

            split_groups[split_name] = (
                current_splits[0]
                if len(current_splits) == 1
                else concatenate_datasets(current_splits)
            )

        for split_name, split_dataset in split_groups.items():
            prepared_splits.setdefault(split_name, []).append(split_dataset)

    combined = DatasetDict(
        {
            split_name: (
                split_datasets[0]
                if len(split_datasets) == 1
                else concatenate_datasets(split_datasets)
            )
            for split_name, split_datasets in prepared_splits.items()
        }
    )
    combined.save_to_disk(str(output_dir))

    return combined


def load_prepared_nli_data(split="train", data_dir=PROCESSED_DATA_DIR):
    dataset_dict = load_from_disk(str(data_dir))
    return dataset_dict[split]


def load_data(split="train", dataset_name="prepared_nli", data_dir=PROCESSED_DATA_DIR):
    if dataset_name == "prepared_nli":
        return load_prepared_nli_data(split=split, data_dir=data_dir)

    if dataset_name == "glue_mrpc":
        cached_arrow = _cached_glue_arrow(split)
        if cached_arrow is not None:
            return HFDataset.from_file(str(cached_arrow))

    HF_CACHE_DIR.mkdir(exist_ok=True)
    return _load_dataset_with_fallback(dataset_name, split=split, cache_dir=HF_CACHE_DIR)


class SentencePairDataset(Dataset):
    def __init__(
        self,
        data,
        text_key_1="sentence1",
        text_key_2="sentence2",
        label_key=None,
    ):
        self.data = data
        self.text_key_1 = text_key_1
        self.text_key_2 = text_key_2
        self.label_key = label_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        item = {
            "text_1": row[self.text_key_1],
            "text_2": row[self.text_key_2],
        }

        if self.label_key is not None and self.label_key in row:
            item[self.label_key] = row[self.label_key]

        if "label_text" in row:
            item["label_text"] = row["label_text"]

        if "source_dataset" in row:
            item["source_dataset"] = row["source_dataset"]

        return item


class SentencePairCollator:
    def __init__(
        self,
        tokenizer,
        *,
        max_length=128,
        pad_to_multiple_of=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        texts_1 = [item["text_1"] for item in batch]
        texts_2 = [item["text_2"] for item in batch]

        tokenizer_kwargs = {
            "truncation": True,
            "padding": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        if self.pad_to_multiple_of is not None:
            tokenizer_kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of

        encoded_1 = self.tokenizer(texts_1, **tokenizer_kwargs)
        encoded_2 = self.tokenizer(texts_2, **tokenizer_kwargs)

        result = {
            "input_ids_1": encoded_1["input_ids"],
            "attention_mask_1": encoded_1["attention_mask"],
            "input_ids_2": encoded_2["input_ids"],
            "attention_mask_2": encoded_2["attention_mask"],
        }

        if "label" in batch[0]:
            result["label"] = [item["label"] for item in batch]

        return result


# Backward-compatible alias for the older notebook/training code.
QuoraDataset = SentencePairDataset
