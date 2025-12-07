"""
Incident severity prediction service.

This module centralises model loading and inference for all four trained models:
 - baseline_cnn (custom lightweight CNN)
 - transfer_resnet18 (fine-tuned ResNet18 head)
 - resnet_gbt (ResNet feature extractor + HistGradientBoostingClassifier)
 - hog_svm (HOG descriptors + SVM)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401
from sklearn.pipeline import Pipeline  # noqa: F401

try:
    from skimage.feature import hog
except ImportError:  # pragma: no cover - optional dependency
    hog = None  # type: ignore


################################################################################
# Shared data structures
################################################################################


@dataclass
class PredictionResult:
    class_id: int
    class_name: str
    probabilities: Dict[str, float]
    multiplier: float


################################################################################
# Simple CNN definition (must match training architecture)
################################################################################


class SimpleCNN(nn.Module):
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
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


################################################################################
# Predictor implementation
################################################################################


DEFAULT_MULTIPLIERS = {
    "01-minor": 1.2,
    "02-moderate": 1.6,
    "03-severe": 3.0,
}


class IncidentPredictor:
    def __init__(
        self,
        model_name: str,
        model_path: Optional[Path] = None,
        severity_multipliers: Optional[Dict[str, float]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.severity_multipliers = severity_multipliers or DEFAULT_MULTIPLIERS
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        loader = getattr(self, f"_load_{model_name}", None)
        if loader is None:
            raise ValueError(f"Unsupported model '{model_name}'")
        loader(model_path)

    # ------------------------------------------------------------------ loaders
    def _load_baseline_cnn(self, model_path: Optional[Path]) -> None:
        if model_path is None:
            model_path = Path("models/baseline_cnn.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        class_names = checkpoint["class_names"]
        self.class_names = class_names
        self.model = SimpleCNN(num_classes=len(class_names)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.model_type = "torch"

    def _load_transfer_resnet18(self, model_path: Optional[Path]) -> None:
        if model_path is None:
            model_path = Path("models/transfer_resnet18.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        class_names = checkpoint["class_names"]
        self.class_names = class_names
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, len(class_names)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.model_type = "torch"

    def _load_transfer_efficientnet_b0(self, model_path: Optional[Path]) -> None:
        if model_path is None:
            model_path = Path("models/transfer_efficientnet_b0.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        class_names = checkpoint["class_names"]
        self.class_names = class_names
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, len(class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        try:
            size = weights.meta.get("min_size", 224)
            mean = weights.meta["mean"]
            std = weights.meta["std"]
        except (KeyError, AttributeError):
            size = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.model_type = "torch"

    def _load_transfer_resnet50(self, model_path: Optional[Path]) -> None:
        if model_path is None:
            model_path = Path("models/transfer_resnet50.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        class_names = checkpoint["class_names"]
        self.class_names = class_names
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, len(class_names)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.model_type = "torch"

    def _load_transfer_vit(self, model_path: Optional[Path]) -> None:
        if model_path is None:
            model_path = Path("models/transfer_vit.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        class_names = checkpoint["class_names"]
        self.class_names = class_names
        self.model = models.vit_b_16(weights=None)
        in_features = self.model.heads.head.in_features
        self.model.heads = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_features, len(class_names)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.model_type = "torch"

    def _load_resnet_gbt(self, model_path: Optional[Path]) -> None:
        if model_path is None:
            model_path = Path("models/resnet_gbt.pkl")
        payload = joblib.load(model_path)
        self.gbt_model = payload["model"]
        self.class_names = payload["class_names"]
        self.feature_extractor = nn.Sequential(
            *list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-1]
        ).to(self.device)
        self.feature_extractor.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.model_type = "sklearn_resnet"

    def _load_hog_svm(self, model_path: Optional[Path]) -> None:
        if hog is None:
            raise RuntimeError("scikit-image is required for HOG+SVM predictor")
        if model_path is None:
            model_path = Path("models/hog_svm.pkl")
        payload = joblib.load(model_path)
        self.hog_model: Pipeline = payload["model"]
        self.class_names = payload["class_names"]
        params = payload["hog_params"]
        self.hog_params = params
        self.transform = transforms.Compose([
            transforms.Resize((params["image_size"], params["image_size"])),
            transforms.Grayscale(),
        ])
        self.model_type = "sklearn_hog"

    # ---------------------------------------------------------------- inference
    def _prepare_image(self, image: Image.Image | Path | str) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def predict(self, image: Image.Image | Path | str) -> PredictionResult:
        image = self._prepare_image(image)

        if self.model_type == "torch":
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        elif self.model_type == "sklearn_resnet":
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self.feature_extractor(tensor).view(1, -1).cpu().numpy()
            probs = self.gbt_model.predict_proba(feats)[0]
        elif self.model_type == "sklearn_hog":
            # transforms pipeline returns grayscale PIL; convert to numpy
            proc = self.transform(image)
            img_np = np.array(proc)  # shape (H, W)
            params = self.hog_params
            descriptor = hog(
                img_np,
                orientations=params["orientations"],
                pixels_per_cell=(params["pixels_per_cell"], params["pixels_per_cell"]),
                cells_per_block=(params["cells_per_block"], params["cells_per_block"]),
                block_norm="L2-Hys",
                feature_vector=True,
            )
            if hasattr(self.hog_model, "predict_proba"):
                probs = self.hog_model.predict_proba([descriptor])[0]
            else:  # pragma: no cover - fallback for legacy models
                scores = self.hog_model.decision_function([descriptor])[0]
                exp = np.exp(scores - np.max(scores))
                probs = exp / exp.sum()
        else:  # pragma: no cover
            raise RuntimeError(f"Unknown model_type {self.model_type}")

        class_id = int(np.argmax(probs))
        class_name = self.class_names[class_id]
        prob_map = {name: float(probs[idx]) for idx, name in enumerate(self.class_names)}
        multiplier = self.severity_multipliers.get(class_name, 1.0)
        return PredictionResult(
            class_id=class_id,
            class_name=class_name,
            probabilities=prob_map,
            multiplier=multiplier,
        )

    def predict_batch(self, images: Sequence[Path | str]) -> List[PredictionResult]:
        return [self.predict(img) for img in images]


################################################################################
# CLI
################################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description="Run incident severity predictions")
    parser.add_argument("--model", choices=[
        "baseline_cnn", "transfer_resnet18", "transfer_efficientnet_b0",
        "transfer_resnet50", "transfer_vit", "resnet_gbt", "hog_svm"
    ], default="transfer_resnet18")
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--images", type=Path, nargs="+", required=True,
                        help="One or more image paths to score")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()

    predictor = IncidentPredictor(
        model_name=args.model,
        model_path=args.model_path,
        device=args.device,
    )
    for img in args.images:
        result = predictor.predict(img)
        print(f"{img}: class={result.class_name} (multiplier={result.multiplier})")
        print("  probabilities:")
        for cls, prob in result.probabilities.items():
            print(f"    {cls}: {prob:.3f}")


if __name__ == "__main__":
    main()
