import argparse
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from training_logger import RunLogger
from training_config import CONFIG


def create_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    weights = models.ResNet18_Weights.DEFAULT
    try:
        normalize = transforms.Normalize(mean=weights.meta["mean"],
                                         std=weights.meta["std"])
    except KeyError:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    size = CONFIG.img_size

    base_tfms = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ]
    train_ds = datasets.ImageFolder(data_root / "training",
                                    transform=transforms.Compose(base_tfms))
    val_ds = datasets.ImageFolder(data_root / "validation",
                                  transform=transforms.Compose(base_tfms))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


def extract_features(
    loader: DataLoader,
    feature_extractor: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_extractor.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            outputs = feature_extractor(images)
            outputs = outputs.view(outputs.size(0), -1)
            feats.append(outputs.cpu().numpy())
            labels.append(target.numpy())
    features = np.concatenate(feats, axis=0)
    targets = np.concatenate(labels, axis=0)
    return features, targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Gradient Boosted Trees on ResNet features"
    )
    parser.add_argument("--data_root", type=Path, default=Path("data3a"))
    parser.add_argument("--batch_size", type=int, default=CONFIG.feature_batch_size)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--log_dir", type=Path, default=Path("training_logs"))
    args = parser.parse_args()

    logger = RunLogger("resnet_gbt", args.log_dir)

    train_loader, val_loader, class_names = create_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1]).to(device)

    logger.log("[INFO] Extracting ResNet18 features for training split...")
    train_feats, train_labels = extract_features(train_loader, feature_extractor, device)
    logger.log(f"[INFO] Train features shape: {train_feats.shape}")

    logger.log("[INFO] Extracting ResNet18 features for validation split...")
    val_feats, val_labels = extract_features(val_loader, feature_extractor, device)
    logger.log(f"[INFO] Validation features shape: {val_feats.shape}")

    clf = HistGradientBoostingClassifier(
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
    )
    logger.log("[INFO] Training HistGradientBoostingClassifier...")
    clf.fit(train_feats, train_labels)

    val_preds = clf.predict(val_feats)
    val_acc = accuracy_score(val_labels, val_preds)
    logger.log(f"[FINAL] Validation accuracy: {val_acc*100:.2f}%")
    logger.log("Confusion matrix:")
    logger.log(confusion_matrix(val_labels, val_preds))
    logger.log("Classification report:")
    logger.log(classification_report(val_labels, val_preds, target_names=class_names))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "resnet_gbt.pkl"
    saved_params = {
        "data_root": str(args.data_root),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "max_iter": args.max_iter,
    }
    joblib.dump({
        "model": clf,
        "class_names": class_names,
        "feature_extractor": "resnet18_pool",
        "params": saved_params,
    }, ckpt_path)
    logger.log(f"[INFO] Saved GBT model to {ckpt_path}")
    logger.record_summary({
        "best_epoch": "-",
        "best_acc": f"{val_acc*100:.2f}",
        "best_loss": "-",
        "epochs": "-",
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "notes": f"max_depth={args.max_depth}, max_iter={args.max_iter}",
    })


if __name__ == "__main__":
    main()
