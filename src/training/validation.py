"""
src/training/validation.py

Evaluation on a loader: loss, accuracy and ROC-AUC.
The positive class is ai_generated (label 1).
"""
import torch
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool = False, max_batches=None) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    probs, labels = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        labels.append(y.cpu())

    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else float("nan")
    return {
        "loss": total_loss / n,
        "accuracy": float((preds == labels).mean()),
        "auc": auc,
    }