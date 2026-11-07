"""tests/test_model.py  (uses pretrained=False, so no download is needed)"""
import torch
import torch.nn as nn

from src.models.efficientnet import (build_efficientnet_b0, count_parameters,
                                     set_backbone_trainable, set_frozen_bn_eval,
                                     unfreeze_last_blocks)
from src.models.model_loader import load_checkpoint, save_checkpoint


def test_forward_shape():
    model = build_efficientnet_b0(num_classes=2, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)


def test_freeze_backbone_leaves_head_trainable():
    model = build_efficientnet_b0(pretrained=False)
    set_backbone_trainable(model, False)
    assert not any(p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    trainable, total = count_parameters(model)
    assert 0 < trainable < total


def test_unfreeze_last_blocks():
    model = build_efficientnet_b0(pretrained=False)
    unfreeze_last_blocks(model, 4)
    blocks = list(model.features.children())
    for i, block in enumerate(blocks):
        expected = i >= len(blocks) - 4
        assert all(p.requires_grad == expected for p in block.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_frozen_batchnorm_stays_in_eval():
    model = build_efficientnet_b0(pretrained=False)
    unfreeze_last_blocks(model, 2)
    model.train()
    set_frozen_bn_eval(model)
    early = [m for m in model.features[1].modules() if isinstance(m, nn.BatchNorm2d)]
    late = [m for m in model.features[-1].modules() if isinstance(m, nn.BatchNorm2d)]
    assert early and not any(m.training for m in early)
    assert late and all(m.training for m in late)


def test_checkpoint_roundtrip(tmp_path):
    model = build_efficientnet_b0(pretrained=False).eval()
    path = tmp_path / "ckpt.pth"
    save_checkpoint(path, model, epoch=3, metrics={"loss": 0.5, "accuracy": 0.9, "auc": 0.95})
    loaded, ckpt = load_checkpoint(path)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)
    assert ckpt["epoch"] == 3
    assert ckpt["class_names"] == ["real", "ai_generated"]
    assert ckpt["img_size"] == 224