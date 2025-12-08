import torch
import warnings
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from training.train_transfer_cnn import build_model, create_dataloaders, get_config

# Suppress warnings
warnings.filterwarnings("ignore")

# Config
DATA_ROOT = Path("data3a")
MODELS_DIR = Path("models/new_models")
OUTPUT_DIR = Path("test_results/validation_metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "EfficientNet-B0": ("transfer_efficientnet_b0", "transfer_efficientnet_b0.pth"),
    "ResNet18": ("transfer_resnet18", "transfer_resnet18.pth"),
    "MobileNet-V3": ("transfer_mobilenet_v3_small", "transfer_mobilenet_v3_small.pth"),
}

@torch.no_grad()
def evaluate_model(name, model_key, filename):
    print(f"Evaluating {name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Config & Data
    # Note: efficientnet_b0 config key is just "efficientnet_b0" in training_config.py
    # But train_transfer_cnn uses "resnet18" or "mobilenet_v3_small"
    # We need to handle the config retrieval carefully or just hardcode image size since we know it's 224
    img_size = 224 
    
    # We can reuse create_dataloaders from train_transfer_cnn for simplicity
    # It returns train_loader, val_loader, class_names
    # We need to mock 'args' or just call it directly
    
    # Build Model
    if "efficientnet" in model_key:
        from torchvision import models
        import torch.nn as nn
        model = models.efficientnet_b0(weights=None)
        # We need to know num_classes first, which we get from data
        # Let's load data first
        pass
    
    # Re-using the logic manually to ensure correctness
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_ds = datasets.ImageFolder(DATA_ROOT / "validation", transform=val_tfms)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    class_names = val_ds.classes
    
    # Init Model Architecture
    import torch.nn as nn
    from torchvision import models
    
    if "efficientnet" in model_key:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(class_names))
    elif "mobilenet" in model_key:
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(class_names))
    else: # resnet
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, len(class_names)),
        )
        
    # Load Weights
    ckpt_path = MODELS_DIR / filename
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Inference
    all_preds = []
    all_targets = []
    
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(labels.tolist())
        
    # Report
    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names)
    
    output_file = OUTPUT_DIR / f"{name}_report.txt"
    with open(output_file, "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n\n")
        f.write(report)
        
    print(f"Finished {name} - Acc: {acc*100:.2f}%")

if __name__ == "__main__":
    for name, (key, fname) in MODELS.items():
        evaluate_model(name, key, fname)
