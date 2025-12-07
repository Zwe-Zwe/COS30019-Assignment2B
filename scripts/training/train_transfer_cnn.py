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
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

from training_logger import RunLogger
from training_config import CONFIG

def default_workers() -> int:
    try:
        import os
        cpus = os.cpu_count() or 2
        return max(2, cpus // 2)
    except Exception:
        return 2


def build_model(num_classes: int,
                model_name: str,
                freeze_backbone: bool = True) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    # Default: resnet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def create_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    strong_aug: bool,
    model_name: str,
    prefetch_factor: int,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
    else:
        weights = models.ResNet18_Weights.DEFAULT
    try:
        normalize = transforms.Normalize(mean=weights.meta["mean"],
                                         std=weights.meta["std"])
        size = weights.meta["min_size"]
    except KeyError:
        # Older torchvision builds may not expose meta info; fallback defaults.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        size = 224

    train_tfms_list = [
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ]
    if strong_aug:
        train_tfms_list.append(transforms.RandomRotation(15))
        train_tfms_list.append(transforms.RandomPerspective(distortion_scale=0.25, p=0.4))
    train_tfms_list.append(transforms.ToTensor())
    if strong_aug:
        train_tfms_list.append(transforms.RandomErasing(p=0.25))
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

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    # Remove None value because DataLoader rejects it.
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, train_ds.classes


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool):
    model.eval()
    total_loss = 0.0
    correct = 0
    preds = []
    targets = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with amp.autocast(enabled=use_amp):
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
        description="Transfer learning using pretrained ResNet18"
    )
    parser.add_argument("--data_root", type=Path, default=Path("data3a"))
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--batch_size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=CONFIG.weight_decay)
    parser.add_argument("--num_workers", type=int, default=default_workers())
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Prefetch factor for each worker (only if num_workers > 0)")
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone weights (default: False, i.e., unfreeze)")
    parser.add_argument("--strong_aug", action="store_true",
                        help="Enable stronger data augmentation")
    parser.add_argument("--no-strong-aug", action="store_false", dest="strong_aug",
                        help="Disable stronger augmentation")
    parser.add_argument("--scheduler", choices=["none", "cosine", "step"],
                        default=CONFIG.scheduler)
    parser.add_argument("--step_size", type=int, default=CONFIG.step_size)
    parser.add_argument("--step_gamma", type=float, default=CONFIG.step_gamma)
    parser.add_argument("--early_stop_patience", type=int,
                        default=CONFIG.early_stop_patience)
    parser.add_argument("--log_dir", type=Path, default=Path("training_logs"),
                        help="Directory to store per-run logs and history")
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v3_small"],
                        default="resnet18", help="Backbone to use (mobilenet is faster/lighter)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable mixed precision training for speed/VRAM savings")
    parser.set_defaults(strong_aug=CONFIG.strong_augmentation)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    model_key = f"transfer_{args.model}"
    logger = RunLogger(model_key, args.log_dir)

    train_loader, val_loader, class_names = create_dataloaders(
        args.data_root, args.batch_size, args.num_workers, args.strong_aug, args.model, args.prefetch_factor
    )
    model = build_model(len(class_names), args.model, freeze_backbone=args.freeze_backbone)
    model.to(device)

    params = model.parameters() if not args.freeze_backbone else model.fc.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = amp.GradScaler(enabled=use_amp)
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
    ckpt_path = args.output_dir / f"{model_key}.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
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
    val_loss, val_acc, preds, targets = evaluate(model, val_loader, criterion, device, use_amp)
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
        "notes": (
            f"model={args.model}; unfreeze={not args.freeze_backbone}; scheduler={args.scheduler}; "
            f"strong_aug={args.strong_aug}; wd={args.weight_decay}; "
            f"early_stop={args.early_stop_patience}; amp={use_amp}; "
            f"workers={args.num_workers}; prefetch={args.prefetch_factor}"
        ),
    })


if __name__ == "__main__":
    main()
