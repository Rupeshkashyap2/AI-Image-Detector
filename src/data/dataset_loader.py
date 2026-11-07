"""
src/data/dataset_loader.py

PyTorch Dataset and DataLoaders for dataset/processed.
Class mapping is explicit: real = 0, ai_generated = 1 (ai_generated is the positive class).

Self-test (from the project root):
    python -m src.data.dataset_loader --workers 2
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentation import get_train_transform
from src.data.preprocessing import get_eval_transform, load_rgb_image

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "dataset" / "processed"

CLASS_NAMES = ["real", "ai_generated"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
SPLITS = ("train", "validation", "test")
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class FaceDataset(Dataset):
    """Reads <split_dir>/real/* and <split_dir>/ai_generated/*."""

    def __init__(self, split_dir, transform, return_path: bool = False):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.return_path = return_path
        self.samples = []
        for name in CLASS_NAMES:
            class_dir = self.split_dir / name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXT)
            self.samples += [(p, CLASS_TO_IDX[name]) for p in files]
        if not self.samples:
            raise RuntimeError(f"No images found under {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = self.transform(load_rgb_image(path))
        if self.return_path:
            return x, label, str(path)
        return x, label

    def class_counts(self) -> dict:
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts


def create_dataloaders(processed_dir=PROCESSED_DIR, batch_size: int = 32,
                       num_workers: int = 4, seed: int = 42) -> dict:
    processed_dir = Path(processed_dir)
    pin_memory = torch.cuda.is_available()
    loaders = {}
    for split in SPLITS:
        is_train = split == "train"
        dataset = FaceDataset(
            processed_dir / split,
            get_train_transform() if is_train else get_eval_transform(),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=is_train,  # avoids a size-1 last batch breaking BatchNorm
            generator=torch.Generator().manual_seed(seed) if is_train else None,
        )
    return loaders


if __name__ == "__main__":
    from torchvision.utils import save_image

    from src.data.preprocessing import denormalize

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    loaders = create_dataloaders(batch_size=args.batch_size, num_workers=args.workers)
    for split, loader in loaders.items():
        print(f"{split:10s} images={len(loader.dataset):5d}  batches={len(loader):4d}  "
              f"classes={loader.dataset.class_counts()}")

    x, y = next(iter(loaders["train"]))
    print(f"\nTrain batch: x{tuple(x.shape)} {x.dtype}, y{tuple(y.shape)} {y.dtype}")
    print(f"  value range: {x.min():.2f} to {x.max():.2f}")
    print(f"  per-channel mean: {[round(v, 2) for v in x.mean(dim=(0, 2, 3)).tolist()]}")
    print(f"  labels in this batch: real={(y == 0).sum().item()}, ai_generated={(y == 1).sum().item()}")

    out = ROOT / "results" / "graphs" / "train_batch_samples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = min(16, len(x))
    save_image(denormalize(x[:n]), out, nrow=8, padding=2)
    print(f"\nSaved sample grid: {out}")
    print(f"Grid labels (row by row, 0=real, 1=ai_generated): {y[:n].tolist()}")