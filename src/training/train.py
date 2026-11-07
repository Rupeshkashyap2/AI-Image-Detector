"""
src/training/train.py

Phase 3: two-stage training of EfficientNetB0.
  Stage 1 (head):     backbone frozen, only the new classifier trains.
  Stage 2 (finetune): last N backbone blocks are unfrozen.
The best model (lowest validation loss) is saved to saved_models/.
The test set is NOT used here.

Run from the project root:
    python -m src.training.train --smoke-test --workers 0   # quick pipeline check
    python -m src.training.train                            # full training
"""
import argparse
import csv
import json
import math
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.data.dataset_loader import CLASS_NAMES, create_dataloaders
from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from src.models.efficientnet import (count_parameters, set_backbone_trainable,
                                     set_frozen_bn_eval, unfreeze_last_blocks)
from src.models.model_loader import build_model, load_checkpoint, save_checkpoint
from src.training.config import GRAPHS_DIR, PROCESSED_DIR, SAVED_MODELS_DIR, TrainConfig
from src.training.loss import get_loss
from src.training.validation import evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, max_batches=None):
    model.train()
    set_frozen_bn_eval(model)
    total_loss, correct, n = 0.0, 0, 0
    for i, (x, y) in enumerate(tqdm(loader, desc="  train", leave=False)):
        if max_batches and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def plot_history(history: list, path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in ((axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], marker="o", label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], marker="o", label="validation")
        first_ft = next((h["epoch"] for h in history if h["stage"] == "finetune"), None)
        if first_ft is not None and first_ft > 1:
            ax.axvline(first_ft - 0.5, color="gray", linestyle="--", label="fine-tuning starts")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def clean(obj):
    """Make values JSON-safe (NaN -> None)."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    return obj


def parse_args(cfg: TrainConfig):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--workers", type=int, default=cfg.num_workers)
    ap.add_argument("--head-epochs", type=int, default=cfg.head_epochs)
    ap.add_argument("--finetune-epochs", type=int, default=cfg.finetune_epochs)
    ap.add_argument("--head-lr", type=float, default=cfg.head_lr)
    ap.add_argument("--lr", type=float, default=cfg.finetune_lr, help="fine-tuning learning rate")
    ap.add_argument("--unfreeze-blocks", type=int, default=cfg.unfreeze_blocks)
    ap.add_argument("--patience", type=int, default=cfg.early_stop_patience)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--smoke-test", action="store_true",
                    help="1 epoch per stage, 20 batches each; writes separate *_smoke files")
    return ap.parse_args()


def main() -> None:
    cfg = TrainConfig()
    args = parse_args(cfg)
    cfg.batch_size, cfg.num_workers = args.batch_size, args.workers
    cfg.head_epochs, cfg.finetune_epochs = args.head_epochs, args.finetune_epochs
    cfg.head_lr, cfg.finetune_lr = args.head_lr, args.lr
    cfg.unfreeze_blocks, cfg.early_stop_patience = args.unfreeze_blocks, args.patience
    cfg.seed = args.seed
    cfg.use_amp = cfg.use_amp and not args.no_amp
    if args.smoke_test:
        cfg.head_epochs, cfg.finetune_epochs, cfg.max_batches = 1, 1, 20
        cfg.run_name = "efficientnet_smoke"

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} | AMP: {use_amp}")
    if device.type == "cpu":
        print("WARNING: training on CPU will be very slow. Check torch.cuda.is_available().")

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = SAVED_MODELS_DIR / f"{cfg.run_name}_best.pth"
    config_path = SAVED_MODELS_DIR / ("model_config_smoke.json" if args.smoke_test else "model_config.json")
    history_path = GRAPHS_DIR / f"{cfg.run_name}_history.csv"
    curves_path = GRAPHS_DIR / f"{cfg.run_name}_curves.png"
    max_batches = cfg.max_batches or None

    loaders = create_dataloaders(PROCESSED_DIR, cfg.batch_size, cfg.num_workers, cfg.seed)
    print(f"Train: {len(loaders['train'].dataset)} | Validation: {len(loaders['validation'].dataset)} images "
          f"| batch size {cfg.batch_size}")

    model = build_model(cfg.arch, cfg.pretrained, cfg.dropout, cfg.num_classes).to(device)
    criterion = get_loss(cfg.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = []
    best_val_loss, best_epoch = float("inf"), 0
    global_epoch = 0
    stages = [("head", cfg.head_epochs, cfg.head_lr), ("finetune", cfg.finetune_epochs, cfg.finetune_lr)]

    for stage, n_epochs, lr in stages:
        if n_epochs <= 0:
            continue
        if stage == "head":
            set_backbone_trainable(model, False)
        else:
            unfreeze_last_blocks(model, cfg.unfreeze_blocks)
        trainable, total = count_parameters(model)
        print(f"\n=== Stage: {stage} | {n_epochs} epochs | lr {lr} | "
              f"trainable params {trainable:,} of {total:,} ===")

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=lr, weight_decay=cfg.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=cfg.scheduler_factor,
                                      patience=cfg.scheduler_patience)
        bad_epochs = 0

        for _ in range(n_epochs):
            global_epoch += 1
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer,
                                              scaler, device, use_amp, max_batches)
            val = evaluate(model, loaders["validation"], criterion, device, use_amp, max_batches)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val["loss"])

            history.append({
                "epoch": global_epoch, "stage": stage, "lr": current_lr,
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": val["loss"], "val_acc": val["accuracy"], "val_auc": val["auc"],
            })

            improved = val["loss"] < best_val_loss - 1e-4
            if improved:
                best_val_loss, best_epoch, bad_epochs = val["loss"], global_epoch, 0
                save_checkpoint(ckpt_path, model, global_epoch, val, cfg.arch, cfg.dropout)
            elif stage == "finetune":
                bad_epochs += 1

            print(f"Epoch {global_epoch:2d} [{stage:8s}] "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {val['loss']:.4f} acc {val['accuracy']:.4f} auc {val['auc']:.4f} | "
                  f"lr {current_lr:.2e} | {time.time() - t0:.0f}s"
                  f"{'  <- best, saved' if improved else ''}")

            if stage == "finetune" and bad_epochs >= cfg.early_stop_patience:
                print(f"Early stopping: no validation improvement for {bad_epochs} fine-tune epochs.")
                break

    if not ckpt_path.exists():
        raise SystemExit("No checkpoint was saved (zero epochs were run).")

    # Reload the best checkpoint and report its validation metrics.
    best_model, ckpt = load_checkpoint(ckpt_path, device)
    final = evaluate(best_model, loaders["validation"], criterion, device, use_amp, max_batches)
    print(f"\nBest epoch: {best_epoch}  |  validation: loss {final['loss']:.4f}, "
          f"accuracy {final['accuracy']:.4f}, AUC {final['auc']:.4f}")
    print(f"Checkpoint: {ckpt_path}")

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, curves_path)

    model_config = {
        "architecture": cfg.arch,
        "checkpoint": ckpt_path.name,
        "class_names": list(CLASS_NAMES),
        "positive_class": "ai_generated",
        "img_size": IMG_SIZE,
        "normalize_mean": list(IMAGENET_MEAN),
        "normalize_std": list(IMAGENET_STD),
        "gradcam_target_layer": "features[-1]",
        "best_epoch": best_epoch,
        "best_validation_metrics": final,
        "epochs_run": len(history),
        "smoke_test": bool(args.smoke_test),
        "training_config": cfg.as_dict(),
    }
    with open(config_path, "w") as f:
        json.dump(clean(model_config), f, indent=2)
    print(f"Saved: {config_path}\n       {history_path}\n       {curves_path}")


if __name__ == "__main__":
    main()