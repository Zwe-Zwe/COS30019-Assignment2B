import argparse
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

from training_logger import RunLogger
from training_config import get_config


def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def create_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    strong_aug: bool,
    img_size: int,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    size = img_size
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        normalize = transforms.Normalize(mean=weights.meta["mean"],
                                         std=weights.meta["std"])
        size = weights.meta.get("min_size", img_size)
    except Exception:
        pass

    train_tfms_list = [
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    ]
    if strong_aug:
        train_tfms_list.append(transforms.RandomRotation(15))
        train_tfms_list.append(transforms.RandomPerspective(distortion_scale=0.3, p=0.4))
    train_tfms_list.append(transforms.ToTensor())
    if strong_aug:
        train_tfms_list.append(transforms.RandomErasing(p=0.3))
    train_tfms_list.append(normalize)
    train_tfms = transforms.Compose(train_tfms_list)

    val_tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(data_root / "training", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_root / "validation", transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    preds = []
    targets = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        preds.extend(predicted.cpu().tolist())
        targets.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy, preds, targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer learning using EfficientNet-B0"
    )
    parser.add_argument("--data_root", type=Path, default=Path("data3a"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze EfficientNet features")
    parser.add_argument("--strong_aug", action="store_true", default=None,
                        help="Enable stronger augmentation")
    parser.add_argument("--no-strong-aug", action="store_false", dest="strong_aug",
                        help="Disable stronger augmentation")
    parser.add_argument("--scheduler", choices=["none", "cosine", "step"],
                        default=None)
    parser.add_argument("--step_size", type=int, default=None)
    parser.add_argument("--step_gamma", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--log_dir", type=Path, default=Path("training_logs"),
                        help="Directory to store per-run logs and history")
    args = parser.parse_args()

    cfg = get_config("efficientnet_b0")
    if args.epochs is None:
        args.epochs = cfg.epochs
    if args.batch_size is None:
        args.batch_size = cfg.batch_size
    if args.lr is None:
        args.lr = cfg.learning_rate
    if args.weight_decay is None:
        args.weight_decay = cfg.weight_decay
    if args.scheduler is None:
        args.scheduler = cfg.scheduler
    if args.step_size is None:
        args.step_size = cfg.step_size
    if args.step_gamma is None:
        args.step_gamma = cfg.step_gamma
    if args.early_stop_patience is None:
        args.early_stop_patience = cfg.early_stop_patience
    if args.strong_aug is None:
        args.strong_aug = cfg.strong_augmentation

    logger = RunLogger("transfer_efficientnet_b0", args.log_dir)

    train_loader, val_loader, class_names = create_dataloaders(
        args.data_root, args.batch_size, args.num_workers, args.strong_aug, cfg.img_size
    )
    model = build_model(len(class_names), freeze_backbone=args.freeze_backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = model.parameters() if not args.freeze_backbone else model.classifier.parameters()
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma
        )

    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "transfer_efficientnet_b0.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        logger.log(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "class_names": class_names,
                "unfreeze": not args.freeze_backbone,
            }, ckpt_path)
            logger.log(f"[INFO] Saved new best model to {ckpt_path} (acc={best_acc*100:.2f}%)")
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step()

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            logger.log(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_acc, preds, targets = evaluate(model, val_loader, criterion, device)
    logger.log(f"[FINAL] Best checkpoint - val_loss: {val_loss:.4f}, val_acc: {val_acc*100:.2f}%")
    logger.log("Confusion matrix:")
    logger.log(confusion_matrix(targets, preds))
    logger.log("Classification report:")
    logger.log(classification_report(targets, preds, target_names=class_names))
    logger.record_summary({
        "best_epoch": best_epoch,
        "best_acc": f"{best_acc*100:.2f}",
        "best_loss": f"{val_loss:.4f}",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "notes": (
            f"unfreeze={not args.freeze_backbone}; scheduler={args.scheduler}; "
            f"strong_aug={args.strong_aug}; wd={args.weight_decay}; "
            f"early_stop={args.early_stop_patience}"
        ),
    })


if __name__ == "__main__":
    main()
