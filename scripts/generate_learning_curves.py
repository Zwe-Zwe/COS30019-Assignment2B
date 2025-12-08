
import sys
import os
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Input/Output
INPUT_CSV = PROJECT_ROOT / "test_results" / "training_data" / "training_history.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found.")
        return

    # Data Structure: data[model] = { "epochs": [], "train_loss": [], "val_loss": [], "val_acc": [] }
    data = {}
    
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["Model"]
            if model not in data:
                data[model] = {"epochs": [], "train_loss": [], "val_loss": [], "val_acc": []}
            
            data[model]["epochs"].append(int(row["Epoch"]))
            data[model]["train_loss"].append(float(row["Train_Loss"]))
            data[model]["val_loss"].append(float(row["Val_Loss"]))
            data[model]["val_acc"].append(float(row["Val_Acc"]))

    if not data:
        print("No data items found.")
        return

    # Colors
    colors = {
        "EfficientNet-B0": "#4285F4", # Blue
        "ResNet18": "#34A853",        # Green
        "MobileNetV3": "#EA4335"      # Red
    }

    # -------------------------------------------------------
    # Chart 1: Validation Accuracy Comparison
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for model, metrics in data.items():
        color = colors.get(model, 'gray')
        plt.plot(metrics["epochs"], metrics["val_acc"], marker='o', label=model, color=color, linewidth=2)
    
    plt.title("Model Validation Accuracy over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "training_curve.png" # Reusing name requested by user context ("training_curve")
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

    # -------------------------------------------------------
    # Chart 2: Loss Dynamics (Train vs Val) - Subplots
    # -------------------------------------------------------
    num_models = len(data)
    fig, axes = plt.subplots(1, num_models, figsize=(15, 5), sharey=True)
    
    if num_models == 1:
        axes = [axes]

    for idx, (model, metrics) in enumerate(data.items()):
        ax = axes[idx]
        ax.plot(metrics["epochs"], metrics["train_loss"], label="Train Loss", linestyle='-', color='blue')
        ax.plot(metrics["epochs"], metrics["val_loss"], label="Val Loss", linestyle='--', color='orange')
        ax.set_title(model)
        ax.set_xlabel("Epoch")
        if idx == 0:
            ax.set_ylabel("Loss")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
    plt.suptitle("Training vs Validation Loss Dynamics", fontsize=14)
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "loss_dynamics.png"
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
