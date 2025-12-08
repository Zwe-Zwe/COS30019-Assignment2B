import re
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import numpy as np

# Configuration
LOG_DIR = Path("training_logs")
TEST_RESULTS_DIR = Path("test_results")
CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Files to map model names to their latest log files
def get_latest_log(model_prefix):
    logs = list(LOG_DIR.glob(f"{model_prefix}_*.log"))
    if not logs:
        return None
    # Sort by timestamp in filename (assuming format *_YYYYMMDD-HHMMSS.log)
    return sorted(logs)[-1].name

LOG_FILES = {
    "ResNet18": get_latest_log("transfer_resnet18"),
    "MobileNet-V3": get_latest_log("transfer_mobilenet_v3_small"),
    "EfficientNet-B0": get_latest_log("transfer_efficientnet_b0"),
}

PRED_FILES = {
    "ResNet18": "predictions_resnet.txt",
    "MobileNet-V3": "predictions_mobilenet.txt",
    "EfficientNet-B0": "predictions_efficientnet.txt",
}

def parse_log_file(filepath):
    """Extracts epoch data from a log file."""
    epochs = []
    val_accs = []
    val_losses = []
    
    with open(filepath, "r") as f:
        for line in f:
            # Match: Epoch 01/5 - train_loss: 1.0974 - val_loss: 0.7771 - val_acc: 68.93%
            match = re.search(r"Epoch (\d+)/\d+ .* val_loss: ([\d.]+) - val_acc: ([\d.]+)%", line)
            if match:
                epochs.append(int(match.group(1)))
                val_losses.append(float(match.group(2)))
                val_accs.append(float(match.group(3)))
    return epochs, val_losses, val_accs

def plot_training_curves():
    """Generates comparison charts for Accuracy and Loss."""
    plt.figure(figsize=(10, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    for model_name, filename in LOG_FILES.items():
        epochs, _, val_accs = parse_log_file(LOG_DIR / filename)
        plt.plot(epochs, val_accs, marker='o', label=model_name)
    
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    for model_name, filename in LOG_FILES.items():
        epochs, val_losses, _ = parse_log_file(LOG_DIR / filename)
        plt.plot(epochs, val_losses, marker='o', label=model_name)
    
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "training_curve_comparison.png")
    print(f"Generated {CHARTS_DIR / 'training_curve_comparison.png'}")

def parse_predictions(filepath):
    """Parses prediction text file to get class counts and confidence."""
    preds = {} # image_name -> (class, confidence)
    class_counts = defaultdict(int)
    
    with open(filepath, "r") as f:
        current_img = None
        for line in f:
            # test_images/63.jpg: class=00-none (multiplier=1.0)
            img_match = re.match(r"(test_images/.*\.jpg): class=(.*) \(multiplier=.*\)", line)
            if img_match:
                current_img = img_match.group(1).split("/")[-1] # filename
                pred_class = img_match.group(2)
                class_counts[pred_class] += 1
                preds[current_img] = {"class": pred_class}
            
            # 00-none: 0.999
            if current_img:
                prob_match = re.match(r"\s+([\w-]+): ([\d.]+)", line)
                if prob_match:
                    cls, prob = prob_match.groups()
                    if cls == preds[current_img]["class"]:
                        preds[current_img]["conf"] = float(prob)
    
    return preds, class_counts

def plot_prediction_distribution(all_counts):
    """Generates a bar chart of predicted class distributions."""
    classes = sorted(list(set(k for counts in all_counts.values() for k in counts.keys())))
    model_names = list(all_counts.keys())
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    
    for i, model in enumerate(model_names):
        counts = [all_counts[model][cls] for cls in classes]
        plt.bar(x + i*width, counts, width, label=model)
        
    plt.xlabel("Class")
    plt.ylabel("Number of Predictions")
    plt.title("Predicted Class Distribution on Test Set")
    plt.xticks(x + width, classes)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(CHARTS_DIR / "prediction_distribution.png")
    print(f"Generated {CHARTS_DIR / 'prediction_distribution.png'}")

def generate_report(model_preds):
    """Generates a markdown report with stats."""
    
    # Calculate agreement
    images = sorted(list(model_preds["EfficientNet-B0"].keys()))
    total = len(images)
    full_agreement = 0
    
    for img in images:
        classes = [model_preds[m][img]["class"] for m in model_preds]
        if len(set(classes)) == 1:
            full_agreement += 1
            
    agreement_rate = (full_agreement / total) * 100
    
    report = f"""# Test Results Analysis

## 1. Model Training Performance

We compared the training dynamics of **ResNet18**, **MobileNet-V3**, and **EfficientNet-B0** over 5 epochs.

![Training Curves](../charts/training_curve_comparison.png)

### Key Observations
- **Top Performer**: EfficientNet-B0 achieved the highest validation accuracy (~78.6%) and lowest loss.
- **Convergence**: All models showed rapid convergence in the first 3 epochs, validating the effectiveness of Transfer Learning.
- **Stability**: ResNet18 provided a stable baseline, while MobileNet-V3 proved to be a competitive lightweight option (~75.7%).

## 2. Prediction Consistency on Test Set

We ran batch predictions on the `{total}` images in the `test_images/` folder. Since ground truth labels are not available for these unlabelled images, we analyzed the **Consensus** between models.

- **Full Consensus Rate**: **{agreement_rate:.1f}%** (Models agreed on {full_agreement}/{total} images)

![Prediction Distribution](../charts/prediction_distribution.png)

### Model-specific Behavior
"""
    
    for model, preds in model_preds.items():
        avg_conf = np.mean([p["conf"] for p in preds.values()])
        report += f"- **{model}**: Average Confidence = **{avg_conf*100:.1f}%**\n"
        
    report += "\n## 3. Detailed Predictions\n\nSee the raw output files for per-image details:\n"
    report += "- [EfficientNet Predictions](predictions_efficientnet.txt)\n"
    report += "- [ResNet Predictions](predictions_resnet.txt)\n"
    report += "- [MobileNet Predictions](predictions_mobilenet.txt)\n"

    with open(TEST_RESULTS_DIR / "comprehensive_analysis.md", "w") as f:
        f.write(report)
    print(f"Generated {TEST_RESULTS_DIR / 'comprehensive_analysis.md'}")

def main():
    # 1. Training Curves
    plot_training_curves()
    
    # 2. Prediction Analysis
    all_counts = {}
    model_preds = {}
    
    for model, filename in PRED_FILES.items():
        preds, counts = parse_predictions(TEST_RESULTS_DIR / filename)
        all_counts[model] = counts
        model_preds[model] = preds
        
    # 3. Prediction Charts
    plot_prediction_distribution(all_counts)
    
    # 4. Report Generation
    generate_report(model_preds)

if __name__ == "__main__":
    main()
