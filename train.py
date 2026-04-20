import argparse
import hashlib
import json
import re
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import PROCESSED_DATA_DIR
from dataset import SentencePairCollator
from dataset import SentencePairDataset
from dataset import load_data
from model import MiniTransformer
from model import mean_pooling


LOCAL_TOKENIZER_GLOBS = (
    ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*",
    ".cache/huggingface/hub/models--bert-base-uncased/snapshots/*",
)
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


class HashingTokenizer:
    def __init__(
        self,
        vocab_size=30522,
        pad_token_id=0,
        unk_token_id=100,
        cls_token_id=101,
        sep_token_id=102,
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def _token_to_id(self, token):
        if not token:
            return self.unk_token_id

        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        bucket_count = self.vocab_size - (self.sep_token_id + 1)
        return self.sep_token_id + 1 + (int.from_bytes(digest, "big") % bucket_count)

    def _encode(self, text, max_length):
        pieces = re.findall(r"\w+|[^\w\s]", text.lower())
        token_ids = [self.cls_token_id]
        token_ids.extend(self._token_to_id(piece) for piece in pieces)
        token_ids.append(self.sep_token_id)

        token_ids = token_ids[:max_length]
        attention_mask = [1] * len(token_ids)

        return token_ids, attention_mask

    def __call__(
        self,
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
        pad_to_multiple_of=None,
    ):
        if not truncation:
            raise ValueError("HashingTokenizer requires truncation=True.")
        if return_tensors != "pt":
            raise ValueError("HashingTokenizer only supports return_tensors='pt'.")

        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        encoded = [self._encode(item, max_length=max_length) for item in texts]

        if padding in (True, "longest"):
            target_length = max(len(token_ids) for token_ids, _ in encoded)
        elif padding == "max_length":
            target_length = max_length
        else:
            raise ValueError("HashingTokenizer requires dynamic or max_length padding.")

        if pad_to_multiple_of is not None and target_length % pad_to_multiple_of != 0:
            target_length = (
                (target_length + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of
            target_length = min(target_length, max_length)

        batch_input_ids = []
        batch_attention_masks = []

        for token_ids, attention_mask in encoded:
            token_ids = token_ids[:target_length]
            attention_mask = attention_mask[:target_length]

            if len(token_ids) < target_length:
                pad_count = target_length - len(token_ids)
                token_ids = token_ids + [self.pad_token_id] * pad_count
                attention_mask = attention_mask + [0] * pad_count

            batch_input_ids.append(token_ids)
            batch_attention_masks.append(attention_mask)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
        }


def _resolve_local_tokenizer_path():
    for pattern in LOCAL_TOKENIZER_GLOBS:
        for path in sorted(Path.home().glob(pattern), reverse=True):
            if (path / "tokenizer.json").exists() or (path / "vocab.txt").exists():
                return path
    return None


def load_tokenizer():
    local_path = _resolve_local_tokenizer_path()
    if local_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(local_path),
            local_files_only=True,
        )
        return tokenizer, str(local_path)

    cache_dir = Path(__file__).resolve().parent / ".hf_cache"
    cache_dir.mkdir(exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            cache_dir=str(cache_dir),
        )
        return tokenizer, "bert-base-uncased"
    except OSError:
        return HashingTokenizer(), "offline-hashing-tokenizer"


def contrastive_loss(emb1, emb2, temperature=0.05):
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)

    logits = emb1 @ emb2.T
    logits = logits / temperature

    labels = torch.arange(logits.size(0), device=logits.device)

    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.T, labels)

    return (loss1 + loss2) / 2


def parse_args():
    parser = argparse.ArgumentParser(description="Train the mini sentence embedding model.")
    parser.add_argument("--dataset", default="prepared_nli")
    parser.add_argument("--dataset-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    return parser.parse_args()


def build_dataloader(dataset, tokenizer, args, device):
    if args.max_train_samples is not None:
        limit = min(args.max_train_samples, len(dataset))
        dataset = dataset.select(range(limit))

    pair_dataset = SentencePairDataset(dataset)
    pad_to_multiple_of = 8 if device.type == "cuda" else None
    collator = SentencePairCollator(
        tokenizer,
        max_length=args.max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    loader_kwargs = {
        "dataset": pair_dataset,
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": len(pair_dataset) >= args.batch_size,
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    return DataLoader(**loader_kwargs)


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

    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)

    autocast_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.float16))
        if use_amp
        else nullcontext
    )

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_index, batch in enumerate(progress_bar, start=1):
        input_ids_1 = batch["input_ids_1"].to(device, non_blocking=True)
        mask_1 = batch["attention_mask_1"].to(device, non_blocking=True)
        input_ids_2 = batch["input_ids_2"].to(device, non_blocking=True)
        mask_2 = batch["attention_mask_2"].to(device, non_blocking=True)

        with autocast_context():
            out1 = model(input_ids_1, mask_1)
            out2 = model(input_ids_2, mask_2)
            emb1 = mean_pooling(out1, mask_1)
            emb2 = mean_pooling(out2, mask_2)
            raw_loss = contrastive_loss(emb1, emb2)
            loss = raw_loss / grad_accum_steps

        scaler.scale(loss).backward()

        should_step = batch_index % grad_accum_steps == 0
        if should_step:
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


def save_checkpoint(
    *,
    model,
    optimizer,
    epoch,
    tokenizer_source,
    args,
    checkpoint_dir,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"mini_transformer_epoch_{epoch}.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.get_config(),
            "tokenizer_source": tokenizer_source,
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


def main():
    args = parse_args()

    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    tokenizer, tokenizer_source = load_tokenizer()
    hf_data = load_data(
        split=args.split,
        dataset_name=args.dataset,
        data_dir=args.dataset_dir,
    )

    if "is_positive" in hf_data.column_names:
        hf_data = hf_data.filter(
            lambda row: row["is_positive"],
            desc="Keeping entailment pairs for contrastive training",
        )

    if len(hf_data) == 0:
        raise RuntimeError("The training split is empty after preprocessing/filtering.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.disable_amp
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    dataloader = build_dataloader(hf_data, tokenizer, args, device)

    model = MiniTransformer(
        vocab_size=tokenizer.vocab_size,
        max_len=args.max_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(f"Using tokenizer: {tokenizer_source}")
    print(f"Loaded {len(hf_data)} training pairs from {args.dataset}:{args.split}")
    print(f"Training on: {device}")
    print(
        "Effective batch size: "
        f"{args.batch_size * args.grad_accum_steps} "
        f"({args.batch_size} x {args.grad_accum_steps} accumulation)"
    )

    last_checkpoint_path = None

    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
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
                tokenizer_source=tokenizer_source,
                args=args,
                checkpoint_dir=args.checkpoint_dir,
            )
            print(f"Checkpoint saved to {last_checkpoint_path}")


if __name__ == "__main__":
    main()
