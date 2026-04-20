import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from clip_dataset import ImageTextCollator
from clip_dataset import ImageTextPairDataset
from clip_dataset import build_image_transform
from model import ClipStyleEmbeddingModel
from model import MiniTransformer
from train import DEFAULT_CHECKPOINT_DIR
from train import HashingTokenizer
from train import load_tokenizer


DEFAULT_CLIP_CHECKPOINT_DIR = Path(__file__).resolve().parent / "clip_checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CLIP-style image tower on top of the frozen text encoder."
    )
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--text-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CLIP_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--image-backbone", default="resnet18")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def validate_paths(args):
    if not args.train_metadata.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {args.train_metadata}\n"
            "Replace the placeholder example path with a real `.jsonl` file."
        )

    if args.image_root is not None and not args.image_root.exists():
        raise FileNotFoundError(
            f"Image root directory not found: {args.image_root}"
        )


def _legacy_text_config():
    return {
        "vocab_size": 30522,
        "max_len": 128,
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 6,
        "ff_dim": 1536,
    }


def resolve_text_checkpoint_path(path=None):
    if path is not None:
        return Path(path)

    metadata_path = DEFAULT_CHECKPOINT_DIR / "latest_checkpoint.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return Path(metadata["latest_checkpoint"])

    fallback = DEFAULT_CHECKPOINT_DIR / "mini_transformer_epoch_3.pt"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "No text checkpoint found. Pass --text-checkpoint or train the text model first."
    )


def load_frozen_text_backbone(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model_config = checkpoint["model_config"]
        state_dict = checkpoint["model_state_dict"]
        tokenizer_source = checkpoint.get("tokenizer_source")
    else:
        model_config = _legacy_text_config()
        state_dict = checkpoint
        tokenizer_source = None

    text_backbone = MiniTransformer(**model_config)
    text_backbone.load_state_dict(state_dict)
    text_backbone.eval()

    return text_backbone, tokenizer_source, model_config


def load_clip_tokenizer(tokenizer_source):
    if tokenizer_source == "offline-hashing-tokenizer":
        return HashingTokenizer(), "offline-hashing-tokenizer"

    if tokenizer_source is not None:
        tokenizer_path = Path(tokenizer_source)
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True,
            )
            return tokenizer, str(tokenizer_path)

    return load_tokenizer()


def clip_contrastive_loss(text_embeddings, image_embeddings, logit_scale):
    logits = logit_scale * (image_embeddings @ text_embeddings.T)
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


def build_dataloader(dataset, tokenizer, args, device):
    if args.max_train_samples is not None:
        dataset.records = dataset.records[: args.max_train_samples]

    collator = ImageTextCollator(tokenizer, max_length=args.max_length)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": len(dataset) >= args.batch_size,
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    return DataLoader(**loader_kwargs)


def save_checkpoint(
    *,
    model,
    optimizer,
    epoch,
    checkpoint_dir,
    args,
    tokenizer_source,
    text_checkpoint_path,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"clip_style_epoch_{epoch}.pt"

    torch.save(
        {
            "checkpoint_type": "clip_style",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "clip_config": model.get_config(),
            "tokenizer_source": tokenizer_source,
            "text_checkpoint_path": str(text_checkpoint_path),
            "train_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        checkpoint_path,
    )

    metadata_path = checkpoint_dir / "latest_checkpoint.json"
    metadata_path.write_text(
        json.dumps({"latest_checkpoint": str(checkpoint_path)}, indent=2),
        encoding="utf-8",
    )
    return checkpoint_path


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    *,
    grad_accum_steps=1,
    use_amp=False,
    max_batches=None,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    autocast_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.float16))
        if use_amp
        else nullcontext
    )

    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc="CLIP Training")

    for batch_index, batch in enumerate(progress_bar, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)

        with autocast_context():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            raw_loss = clip_contrastive_loss(
                outputs["text_embeddings"],
                outputs["image_embeddings"],
                outputs["logit_scale"],
            )
            loss = raw_loss / grad_accum_steps

        scaler.scale(loss).backward()

        if batch_index % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += raw_loss.item()
        num_batches += 1
        progress_bar.set_postfix(loss=f"{raw_loss.item():.4f}")

        if max_batches is not None and batch_index >= max_batches:
            break

    if num_batches % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(num_batches, 1)


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    validate_paths(args)

    text_checkpoint_path = resolve_text_checkpoint_path(args.text_checkpoint)
    text_backbone, tokenizer_source, _ = load_frozen_text_backbone(text_checkpoint_path)
    tokenizer, tokenizer_source = load_clip_tokenizer(tokenizer_source)

    dataset = ImageTextPairDataset(
        args.train_metadata,
        image_root=args.image_root,
        image_transform=build_image_transform(args.image_size),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.disable_amp
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    dataloader = build_dataloader(dataset, tokenizer, args, device)
    model = ClipStyleEmbeddingModel(
        text_backbone,
        projection_dim=args.projection_dim,
        image_backbone=args.image_backbone,
        freeze_text_backbone=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(f"Loaded frozen text checkpoint: {text_checkpoint_path}")
    print(f"Using tokenizer: {tokenizer_source}")
    print(f"Training image-text pairs: {len(dataset)}")
    print(f"Training on: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    last_checkpoint_path = None
    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            scaler,
            device,
            grad_accum_steps=args.grad_accum_steps,
            use_amp=use_amp,
            max_batches=args.max_batches,
        )
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

        if args.save_every_epoch or epoch == args.epochs - 1:
            last_checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                checkpoint_dir=args.checkpoint_dir,
                args=args,
                tokenizer_source=tokenizer_source,
                text_checkpoint_path=text_checkpoint_path,
            )
            print(f"Checkpoint saved to {last_checkpoint_path}")


if __name__ == "__main__":
    main()
