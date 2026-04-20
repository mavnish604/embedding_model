import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision import models as tv_models


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        token_embeddings = self.token_embed(input_ids)
        position_embeddings = self.pos_embed(positions)

        return token_embeddings + position_embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, channels = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, channels)

        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        *,
        max_len=128,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        ff_dim=1024,
        gradient_checkpointing=False,
    ):
        super().__init__()

        self.config = {
            "vocab_size": vocab_size,
            "max_len": max_len,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ff_dim": ff_dim,
            "gradient_checkpointing": gradient_checkpointing,
        }
        self.gradient_checkpointing = gradient_checkpointing
        self.embed = Embeddings(vocab_size, embed_dim, max_len=max_len)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda hidden_states: layer(hidden_states, attention_mask),
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, attention_mask)

        return x

    def get_config(self):
        return dict(self.config)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.proj(x)


class ResNetImageEncoder(nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super().__init__()

        builders = {
            "resnet18": tv_models.resnet18,
            "resnet34": tv_models.resnet34,
            "resnet50": tv_models.resnet50,
        }
        if backbone_name not in builders:
            raise ValueError(
                f"Unsupported image backbone '{backbone_name}'. "
                f"Expected one of: {sorted(builders)}"
            )

        backbone = builders[backbone_name](weights=None)
        self.feature_dim = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone_name = backbone_name

    def forward(self, pixel_values):
        features = self.backbone(pixel_values)
        return features.flatten(start_dim=1)


class ClipStyleEmbeddingModel(nn.Module):
    def __init__(
        self,
        text_backbone,
        *,
        projection_dim=256,
        image_backbone="resnet18",
        freeze_text_backbone=True,
    ):
        super().__init__()

        if not hasattr(text_backbone, "config"):
            raise ValueError("text_backbone must expose a config dict")

        self.text_backbone = text_backbone
        self.text_backbone_is_frozen = freeze_text_backbone
        text_embed_dim = text_backbone.get_config()["embed_dim"]

        self.text_projection = ProjectionHead(text_embed_dim, projection_dim)
        self.image_encoder = ResNetImageEncoder(backbone_name=image_backbone)
        self.image_projection = ProjectionHead(
            self.image_encoder.feature_dim,
            projection_dim,
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

        if self.text_backbone_is_frozen:
            self.freeze_text_backbone()

        self.config = {
            "projection_dim": projection_dim,
            "image_backbone": image_backbone,
            "freeze_text_backbone": freeze_text_backbone,
            "text_backbone_config": text_backbone.get_config(),
        }

    def freeze_text_backbone(self):
        self.text_backbone.eval()
        for parameter in self.text_backbone.parameters():
            parameter.requires_grad = False

    def encode_text_features(self, input_ids, attention_mask):
        if self.text_backbone_is_frozen:
            with torch.no_grad():
                hidden_states = self.text_backbone(input_ids, attention_mask)
                pooled = mean_pooling(hidden_states, attention_mask)
        else:
            hidden_states = self.text_backbone(input_ids, attention_mask)
            pooled = mean_pooling(hidden_states, attention_mask)

        return pooled

    def encode_text(self, input_ids, attention_mask):
        text_features = self.encode_text_features(input_ids, attention_mask)
        projected = self.text_projection(text_features)
        return torch.nn.functional.normalize(projected, dim=1)

    def encode_image_features(self, pixel_values):
        return self.image_encoder(pixel_values)

    def encode_image(self, pixel_values):
        image_features = self.encode_image_features(pixel_values)
        projected = self.image_projection(image_features)
        return torch.nn.functional.normalize(projected, dim=1)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_embeddings = self.encode_text(input_ids, attention_mask)
        image_embeddings = self.encode_image(pixel_values)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        return {
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "logit_scale": logit_scale,
        }

    def train(self, mode=True):
        super().train(mode)
        if self.text_backbone_is_frozen:
            self.text_backbone.eval()
        return self

    def get_config(self):
        return dict(self.config)


def mean_pooling(x, mask):
    mask = mask.unsqueeze(-1).to(dtype=x.dtype)
    summed = torch.sum(x * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts
