"""
Dataset preprocessing utility.

Usage:
    python scripts/data_prep/preprocess_dataset.py \
        --raw_dir data_raw \
        --output_dir data3a \
        --train_ratio 0.7 --val_ratio 0.15 \
        --img_size 256 --seed 42
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def gather_images(raw_dir: Path) -> Dict[str, List[Path]]:
    """Return mapping class_name -> list of image paths."""
    class_map: Dict[str, List[Path]] = {}
    for cls_dir in raw_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if images:
            class_map[cls_dir.name] = sorted(images)
    if not class_map:
        raise RuntimeError(f"No class folders with images found in {raw_dir}")
    return class_map


def stratified_split(items: Sequence[Path], train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_train = int(len(items) * train_ratio)
    n_val = int(len(items) * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    to_list = lambda ids: [items[i] for i in ids]
    return to_list(train_idx), to_list(val_idx), to_list(test_idx)


def copy_images(images: List[Path], dest_dir: Path, img_size: int | None):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in images:
        dst = dest_dir / src.name
        if img_size:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im = im.resize((img_size, img_size))
                im.save(dst)
        else:
            shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw accident dataset into train/val/test splits.")
    parser.add_argument("--raw_dir", type=Path, required=True, help="Directory containing class subfolders of raw images")
    parser.add_argument("--output_dir", type=Path, default=Path("data3a"),
                        help="Root directory to create training/validation/test subfolders")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--img_size", type=int, default=None,
                        help="Optional square size to resize images; omit to keep original resolution")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    class_images = gather_images(args.raw_dir)
    test_ratio = 1.0 - (args.train_ratio + args.val_ratio)
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 to leave room for test split")

    print(f"Found classes: {list(class_images.keys())}")
    for split in ["training", "validation", "test"]:
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)

    summary = []
    for cls, imgs in class_images.items():
        train, val, test = stratified_split(imgs, args.train_ratio, args.val_ratio, args.seed)
        copy_images(train, args.output_dir / "training" / cls, args.img_size)
        copy_images(val, args.output_dir / "validation" / cls, args.img_size)
        copy_images(test, args.output_dir / "test" / cls, args.img_size)
        summary.append((cls, len(train), len(val), len(test)))

    print("Split summary (train/val/test):")
    for cls, tr, va, te in summary:
        print(f"  {cls}: {tr}/{va}/{te}")


if __name__ == "__main__":
    main()
