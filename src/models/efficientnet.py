"""
src/models/efficientnet.py

EfficientNetB0 with a 2-class head, plus helpers for freezing and unfreezing.
Structure: model.features (9 blocks) -> model.avgpool (global average pooling)
-> model.classifier (Dropout + Linear).
"""
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True,
                          dropout: float = 0.2) -> nn.Module:
    """Pretrained weights (ImageNet) are downloaded on first use, about 20 MB."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> None:
    """Freeze the whole backbone, then unfreeze only the last n_blocks of model.features."""
    set_backbone_trainable(model, False)
    if n_blocks <= 0:
        return
    for block in list(model.features.children())[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


def set_frozen_bn_eval(model: nn.Module) -> None:
    """Call right after model.train(): keep BatchNorm layers that have no trainable
    parameters in eval mode, so frozen layers do not drift."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def count_parameters(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total