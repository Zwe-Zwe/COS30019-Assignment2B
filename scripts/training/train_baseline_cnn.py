import argparse
import os
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
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from training_logger import RunLogger
from training_config import CONFIG

class SimpleCNN(nn.Module):
    """Lightweight baseline CNN with adaptive pooling for arbitrary image size."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def create_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    img_size: int,
    strong_aug: bool,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tfms_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ]
    if strong_aug:
        train_tfms_list.insert(1, transforms.RandomRotation(12))
        train_tfms_list.insert(2, transforms.RandomPerspective(distortion_scale=0.2, p=0.3))
    train_tfms_list.append(transforms.ToTensor())
    if strong_aug:
        train_tfms_list.append(transforms.RandomErasing(p=0.25))
    train_tfms_list.append(normalize)
    train_tfms = transforms.Compose(train_tfms_list)
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, list[int], list[int]]:
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
        description="Train baseline CNN for traffic incident severity classification"
    )
    parser.add_argument("--data_root", type=Path, default=Path("data3a"),
                        help="Root directory containing training/ and validation/ subdirectories")
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--batch_size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=CONFIG.weight_decay)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=CONFIG.img_size)
    parser.add_argument("--strong_aug", action="store_true",
                        help="Enable stronger spatial augmentations")
    parser.add_argument("--no-strong-aug", action="store_false", dest="strong_aug",
                        help="Disable stronger spatial augmentations")
    parser.add_argument("--scheduler", choices=["none", "cosine", "step"],
                        default=CONFIG.scheduler)
    parser.add_argument("--step_size", type=int, default=CONFIG.step_size,
                        help="Epoch interval for step scheduler")
    parser.add_argument("--step_gamma", type=float, default=CONFIG.step_gamma,
                        help="Decay factor for step scheduler")
    parser.add_argument("--early_stop_patience", type=int,
                        default=CONFIG.early_stop_patience,
                        help="Stop if no val improvement for N epochs (0=off)")
    parser.add_argument("--output_dir", type=Path, default=Path("models"),
                        help="Where to save the best model checkpoint")
    parser.add_argument("--log_dir", type=Path, default=Path("training_logs"),
                        help="Directory to store per-run logs and history")
    parser.set_defaults(strong_aug=CONFIG.strong_augmentation)
    args = parser.parse_args()

    logger = RunLogger("baseline_cnn", args.log_dir)

    train_loader, val_loader, class_names = create_dataloaders(
        args.data_root, args.batch_size, args.num_workers,
        args.img_size, args.strong_aug
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    ckpt_path = args.output_dir / "baseline_cnn.pth"

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
            }, ckpt_path)
            logger.log(f"[INFO] Saved new best model to {ckpt_path} (acc={best_acc*100:.2f}%)")
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step()

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            logger.log(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

    # Final evaluation with metrics
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
        "notes": (
            f"scheduler={args.scheduler}; strong_aug={args.strong_aug}; "
            f"wd={args.weight_decay}; early_stop={args.early_stop_patience}"
        ),
    })


if __name__ == "__main__":
    main()
