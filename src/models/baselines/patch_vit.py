"""Baseline 0 model: ImageNet-pretrained ViT-S/16, fine-tuned (spec-locked).

Fixed backbone for the whole ladder so the primary comparison isolates
aggregation / relational reasoning rather than the visual representation
(``docs/baseline0_spec.md``). Exposes:
  * ``forward`` -> class logits (binary);
  * ``embed``   -> pooled pre-logit features (for caching + later graph rungs);
  * ``data_config`` / ``build_transforms`` -> the exact timm preprocessing;
  * ``param_counts`` -> total / trainable params for the reproducibility record.

Torch/timm are imported at module load; this module is only used in the
training environment (Colab), not in the metadata/EDA path.
"""

from __future__ import annotations

from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn

DEFAULT_BACKBONE = "vit_small_patch16_224"


class PatchViT(nn.Module):
    """timm ViT backbone + linear classification head."""

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.pretrained = pretrained
        # num_classes set on the timm model so it builds the matching head.
        self.model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
        self.embed_dim = int(getattr(self.model, "num_features", 0))

        if freeze_backbone:
            for name, p in self.model.named_parameters():
                if not name.startswith("head"):
                    p.requires_grad_(False)
        self.freeze_backbone = freeze_backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled pre-logit embeddings (batch, embed_dim)."""
        feats = self.model.forward_features(x)
        return self.model.forward_head(feats, pre_logits=True)

    # --- reproducibility / preprocessing helpers ---------------------------

    def data_config(self) -> Dict:
        """Resolve the timm preprocessing config (input size, mean, std, ...)."""
        return timm.data.resolve_data_config({}, model=self.model)

    def build_transforms(self, is_training: bool):
        """Build the timm transform matching this backbone's pretrained cfg."""
        cfg = self.data_config()
        return timm.data.create_transform(**cfg, is_training=is_training)

    def param_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_params": int(total), "trainable_params": int(trainable)}


def build_model(cfg: Dict) -> Tuple[PatchViT, Dict]:
    """Instantiate PatchViT from a config dict; return (model, info-for-logging)."""
    model = PatchViT(
        backbone=cfg.get("backbone", DEFAULT_BACKBONE),
        num_classes=cfg.get("num_classes", 2),
        pretrained=cfg.get("pretrained", True),
        freeze_backbone=cfg.get("freeze_backbone", False),
    )
    info = {
        "backbone": model.backbone_name,
        "pretrained": model.pretrained,
        "freeze_backbone": model.freeze_backbone,
        "embed_dim": model.embed_dim,
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "data_config": model.data_config(),
        **model.param_counts(),
    }
    return model, info
