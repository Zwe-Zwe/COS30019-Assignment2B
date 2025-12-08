"""
Incident severity prediction service.

This module centralises model loading and inference for the supported models:
 - transfer_efficientnet_b0
 - transfer_resnet18
 - transfer_mobilenet_v3_small
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models


################################################################################
# Shared data structures
################################################################################


@dataclass
class PredictionResult:
    class_id: int
    class_name: str
    probabilities: Dict[str, float]
    multiplier: float


DEFAULT_MULTIPLIERS = {
    "00-none": 1.0,
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

    def _load_transfer_mobilenet_v3_small(self, model_path: Optional[Path]) -> None:
        if model_path is None:
            model_path = Path("models/transfer_mobilenet_v3_small.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        class_names = checkpoint["class_names"]
        self.class_names = class_names
        self.model = models.mobilenet_v3_small(weights=None)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, len(class_names))
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

    # ---------------------------------------------------------------- inference
    def _prepare_image(self, image: Image.Image | Path | str) -> Image.Image:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def predict(self, image: Image.Image | Path | str) -> PredictionResult:
        image = self._prepare_image(image)

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

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
        "transfer_resnet18", "transfer_efficientnet_b0", "transfer_mobilenet_v3_small"
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
