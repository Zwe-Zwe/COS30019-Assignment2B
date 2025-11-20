import argparse
from pathlib import Path

import joblib
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import datasets, transforms

from training_logger import RunLogger
from training_config import CONFIG


def compute_hog_features(split_dir: Path, image_size: int,
                         orientations: int, pixels_per_cell: int,
                         cells_per_block: int):
    tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(split_dir, transform=tfms)
    features = []
    labels = []
    for img_tensor, label in dataset:
        img = img_tensor.squeeze(0).numpy()
        vec = hog(
            img,
            orientations=orientations,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(cells_per_block, cells_per_block),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        features.append(vec)
        labels.append(label)
    return np.array(features), np.array(labels), dataset.classes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train HOG + SVM classifier for incident severity"
    )
    parser.add_argument("--data_root", type=Path, default=Path("data3a"))
    parser.add_argument("--image_size", type=int, default=CONFIG.img_size)
    parser.add_argument("--orientations", type=int, default=CONFIG.hog_orientations)
    parser.add_argument("--pixels_per_cell", type=int, default=CONFIG.hog_pixels_per_cell)
    parser.add_argument("--cells_per_block", type=int, default=CONFIG.hog_cells_per_block)
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--log_dir", type=Path, default=Path("training_logs"))
    args = parser.parse_args()

    logger = RunLogger("hog_svm", args.log_dir)

    logger.log("[INFO] Computing HOG features for training split...")
    train_feats, train_labels, class_names = compute_hog_features(
        args.data_root / "training",
        args.image_size,
        args.orientations,
        args.pixels_per_cell,
        args.cells_per_block,
    )
    logger.log(f"[INFO] Train feature matrix: {train_feats.shape}")

    logger.log("[INFO] Computing HOG features for validation split...")
    val_feats, val_labels, _ = compute_hog_features(
        args.data_root / "validation",
        args.image_size,
        args.orientations,
        args.pixels_per_cell,
        args.cells_per_block,
    )
    logger.log(f"[INFO] Validation feature matrix: {val_feats.shape}")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=args.C, gamma=args.gamma, kernel="rbf", probability=True)),
    ])
    logger.log("[INFO] Training SVM on HOG features...")
    clf.fit(train_feats, train_labels)

    val_preds = clf.predict(val_feats)
    val_acc = accuracy_score(val_labels, val_preds)
    logger.log(f"[FINAL] Validation accuracy: {val_acc*100:.2f}%")
    logger.log("Confusion matrix:")
    logger.log(confusion_matrix(val_labels, val_preds))
    logger.log("Classification report:")
    logger.log(classification_report(val_labels, val_preds, target_names=class_names))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "hog_svm.pkl"
    joblib.dump({
        "model": clf,
        "class_names": class_names,
        "hog_params": {
            "image_size": args.image_size,
            "orientations": args.orientations,
            "pixels_per_cell": args.pixels_per_cell,
            "cells_per_block": args.cells_per_block,
        },
        "svm_params": {
            "C": args.C,
            "gamma": args.gamma,
            "kernel": "rbf",
        },
    }, ckpt_path)
    logger.log(f"[INFO] Saved HOG+SVM model to {ckpt_path}")
    logger.record_summary({
        "best_epoch": "-",
        "best_acc": f"{val_acc*100:.2f}",
        "best_loss": "-",
        "epochs": "-",
        "batch_size": "-",
        "lr": "-",
        "notes": (
            f"hog={args.orientations}/{args.pixels_per_cell}/{args.cells_per_block}; "
            f"C={args.C}, gamma={args.gamma}"
        ),
    })


if __name__ == "__main__":
    main()
