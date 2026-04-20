from pathlib import Path
import json

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer

from clip_dataset import build_image_transform
from model import ClipStyleEmbeddingModel
from model import MiniTransformer
from model import mean_pooling
from train import DEFAULT_CHECKPOINT_DIR
from train import HashingTokenizer
from train import load_tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _legacy_text_config():
    return {
        "vocab_size": 30522,
        "max_len": 128,
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 6,
        "ff_dim": 1536,
    }


def resolve_default_checkpoint():
    clip_latest = Path(__file__).resolve().parent / "clip_checkpoints" / "latest_checkpoint.json"
    if clip_latest.exists():
        clip_metadata = json.loads(clip_latest.read_text(encoding="utf-8"))
        return Path(clip_metadata["latest_checkpoint"])

    text_latest = DEFAULT_CHECKPOINT_DIR / "latest_checkpoint.json"
    if text_latest.exists():
        text_metadata = json.loads(text_latest.read_text(encoding="utf-8"))
        return Path(text_metadata["latest_checkpoint"])

    return DEFAULT_CHECKPOINT_DIR / "mini_transformer_epoch_3.pt"


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=device)

    if checkpoint.get("checkpoint_type") == "clip_style":
        clip_config = checkpoint["clip_config"]
        text_backbone = MiniTransformer(**clip_config["text_backbone_config"])
        model = ClipStyleEmbeddingModel(
            text_backbone=text_backbone,
            projection_dim=clip_config["projection_dim"],
            image_backbone=clip_config["image_backbone"],
            freeze_text_backbone=clip_config.get("freeze_text_backbone", True),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint.get("tokenizer_source"), "clip_style"

    if "model_state_dict" in checkpoint:
        model_config = checkpoint["model_config"]
        state_dict = checkpoint["model_state_dict"]
        tokenizer_source = checkpoint.get("tokenizer_source")
    else:
        model_config = _legacy_text_config()
        state_dict = checkpoint
        tokenizer_source = None

    model = MiniTransformer(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer_source, "text"


def load_checkpoint_tokenizer(tokenizer_source):
    if tokenizer_source == "offline-hashing-tokenizer":
        return HashingTokenizer()

    if tokenizer_source is not None:
        tokenizer_path = Path(tokenizer_source)
        if tokenizer_path.exists():
            return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)

    tokenizer, _ = load_tokenizer()
    return tokenizer


model_path = resolve_default_checkpoint()
model, tokenizer_source, checkpoint_type = load_checkpoint(model_path)
tokenizer = load_checkpoint_tokenizer(tokenizer_source)


def get_text_embedding(text, max_length=96):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    input_ids = tokens["input_ids"].to(device)
    mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        if checkpoint_type == "clip_style":
            emb = model.encode_text(input_ids, mask)
        else:
            out = model(input_ids, mask)
            emb = mean_pooling(out, mask)
            emb = F.normalize(emb, dim=1)

    return emb


def get_image_embedding(image_path, image_size=224):
    if checkpoint_type != "clip_style":
        raise RuntimeError("Loaded checkpoint is text-only. Train/load a clip_style checkpoint first.")

    transform = build_image_transform(image_size)
    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(pixel_values)

    return emb


if __name__ == "__main__":
    pairs = [
        ("I love machine learning", "I enjoy studying AI"),
        ("How to learn Python?", "Best way to study coding"),
        ("The sky is blue", "I ate pizza yesterday"),
        ("He is playing football", "A man is playing a sport"),
        ("I hate exams", "I love exams"),
    ]

    # Text → Text similarity
    e1 = get_text_embedding("a photo of a cat")
    e2 = get_text_embedding("a picture of a dog")
    print(f"Text↔Text (cat vs dog): {F.cosine_similarity(e1, e2).item():.4f}")

    # Image → Text similarity (cross-modal search)
    img_emb = get_image_embedding("data/clip/cifar10_train/images/00000_frog.png")
    txt_emb = get_text_embedding("a photo of a frog")
    print(f"Image↔Text (frog img vs 'frog' text): {F.cosine_similarity(img_emb, txt_emb).item():.4f}")

    # Image → Image similarity
    img1 = get_image_embedding("data/clip/cifar10_train/images/00000_frog.png")
    img2 = get_image_embedding("data/clip/cifar10_train/images/00001_truck.png")
    print(f"Image↔Image (frog vs truck): {F.cosine_similarity(img1, img2).item():.4f}")

    print(f"Loaded checkpoint: {model_path}")
    print(f"Checkpoint type: {checkpoint_type}")

    for a, b in pairs:
        e1 = get_text_embedding(a)
        e2 = get_text_embedding(b)

        sim = F.cosine_similarity(e1, e2)
        print(f"{a} | {b} -> {sim.item():.4f}")
