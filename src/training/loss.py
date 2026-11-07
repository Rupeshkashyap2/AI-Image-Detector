"""
src/training/loss.py

Loss function. Classes are balanced, so no class weights by default.
"""
import torch.nn as nn


def get_loss(label_smoothing: float = 0.0, class_weights=None) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)