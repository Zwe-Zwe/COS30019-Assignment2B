
import sys
import os
import glob
import re
import csv
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "training_logs"
OUTPUT_DIR = PROJECT_ROOT / "test_results" / "training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    "transfer_efficientnet_b0",
    "transfer_resnet18",
    "transfer_mobilenet_v3_small"
]

def parse_log_file(log_path):
    history = []
    best_results = {"val_acc": 0.0, "epoch": 0}
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        
        # Parse Epoch line: "Epoch 01/45 - train_loss: 0.9023 - val_loss: 0.5925 - val_acc: 73.79%"
        epoch_match = re.search(r"Epoch (\d+)/(\d+) - train_loss: ([\d.]+) - val_loss: ([\d.]+) - val_acc: ([\d.]+)%", line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            train_loss = float(epoch_match.group(3))
            val_loss = float(epoch_match.group(4))
            val_acc = float(epoch_match.group(5))
            
            history.append({
                "Epoch": epoch,
                "Train_Loss": train_loss,
                "Val_Loss": val_loss,
                "Val_Acc": val_acc
            })
            
            if val_acc > best_results["val_acc"]:
                best_results["val_acc"] = val_acc
                best_results["epoch"] = epoch
                
    return history, best_results

def generate_report():
    all_history = []
    summary_data = []
    
    print(f"Generating training report in: {OUTPUT_DIR}")
    
    for model_name in MODELS:
        # Find latest log for this model
        pattern = str(LOG_DIR / f"{model_name}_*.log")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"No logs found for {model_name}")
            continue
            
        latest_log = files[-1]
        print(f"Parsing {Path(latest_log).name}...")
        
        history, best = parse_log_file(latest_log)
        
        # Add model name to history
        # Simplify model name for display
        display_name = model_name.replace("transfer_", "").replace("_small", "")
        if display_name == "resnet18": display_name = "ResNet18"
        elif display_name == "efficientnet_b0": display_name = "EfficientNet-B0"
        elif "mobilenet" in display_name: display_name = "MobileNetV3"
        
        for record in history:
            record["Model"] = display_name
            record["Model_Full"] = model_name
            all_history.append(record)
            
        summary_data.append({
            "Model": display_name,
            "Best_Val_Acc": f"{best['val_acc']:.2f}%",
            "Best_Epoch": best["epoch"],
            "Total_Epochs_Run": len(history)
        })

    # Save History CSV
    if all_history:
        history_csv = OUTPUT_DIR / "training_history.csv"
        keys = ["Model", "Model_Full", "Epoch", "Train_Loss", "Val_Loss", "Val_Acc"]
        with open(history_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            # Order records: model, then epoch
            # Assuming all_history is already sorted by model (from loop) then epoch (from parse)
            writer.writerows(all_history)
        print(f"Saved {history_csv}")

    # Save Summary CSV
    if summary_data:
        summary_csv = OUTPUT_DIR / "training_summary.csv"
        keys = ["Model", "Best_Val_Acc", "Best_Epoch", "Total_Epochs_Run"]
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"Saved {summary_csv}")

if __name__ == "__main__":
    generate_report()
