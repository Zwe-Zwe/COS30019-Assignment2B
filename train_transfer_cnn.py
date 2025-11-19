import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
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
) -> Tuple[DataLoader, DataLoader, list[str]]:
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

    train_tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
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
        description="Transfer learning using pretrained ResNet18"
    )
    parser.add_argument("--data_root", type=Path, default=Path("data3a"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=Path("models"))
    parser.add_argument("--unfreeze", action="store_true",
                        help="Fine-tune entire network if set")
    args = parser.parse_args()

    train_loader, val_loader, class_names = create_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )
    model = build_model(len(class_names), freeze_backbone=not args.unfreeze)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = model.parameters() if args.unfreeze else model.fc.parameters()
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "transfer_resnet18.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{args.epochs} "
              f"- train_loss: {train_loss:.4f} "
              f"- val_loss: {val_loss:.4f} "
              f"- val_acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "class_names": class_names,
                "unfreeze": args.unfreeze,
            }, ckpt_path)
            print(f"[INFO] Saved new best model to {ckpt_path} (acc={best_acc*100:.2f}%)")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_acc, preds, targets = evaluate(model, val_loader, criterion, device)
    print(f"[FINAL] Best checkpoint - val_loss: {val_loss:.4f}, val_acc: {val_acc*100:.2f}%")
    print("Confusion matrix:")
    print(confusion_matrix(targets, preds))
    print("Classification report:")
    print(classification_report(targets, preds, target_names=class_names))


if __name__ == "__main__":
    main()
