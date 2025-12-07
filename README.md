# Incident Classification System – COS30019 Assignment 2B

This repository contains both the ML training pipeline and the integrated ICS (Incident Classification System) for the Kuching heritage map. It builds on Part A (search algorithms) and extends it with multiple ML classifiers, a Flask-based GUI, and top‑k routing.

## Quickstart (step by step)
1) **Create venv** (optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) **Install dependencies**
```bash
pip install -r requirements.txt
# If your platform needs a specific PyTorch build, install that first, then re-run pip -r
```

3) **Prepare data**  
If you have raw, classed folders, split them with the helper (writes into `data3a/`):
```bash
python3 scripts/data_prep/preprocess_dataset.py --raw_dir path/to/raw_classes --output_dir data3a --img_size 256
```
Otherwise ensure `data3a/training` and `data3a/validation` exist with class subfolders.

4) **Make sure model weights are present**  
Pretrained checkpoints live in `models/` (`transfer_efficientnet_b0.pth`, `transfer_resnet18.pth`, `transfer_mobilenet_v3_small.pth`). Train from scratch if needed:
```bash
# ResNet18
python3 scripts/training/train_transfer_cnn.py --data_root data3a --unfreeze
# EfficientNet-B0
python3 scripts/training/train_efficientnet_b0.py --data_root data3a --strong_aug
```

5) **Run the GUI (ICS)**  
```bash
FLASK_APP=ics_app.py flask run    # or: python3 ics_app.py
```
Then open http://127.0.0.1:5000, choose start/goal, upload an incident image or set severity + way, pick model, set k routes, and view the highlighted incident + top-k paths.

6) **CLI prediction (no GUI)**  
```bash
python3 incident_predictor.py --model transfer_resnet18 --images path/to/img1.jpg
```

7) **Scripted routing scenarios**  
```bash
python3 scripts/run_scenarios.py --scenarios scenarios/sample_scenarios.json
```

8) **(Optional research) Visual evidence via Flask UI**  
- Run the app, set an incident (way + severity), k=3–5, and capture screenshots of the map/routes.  
- Suggested scenarios and notes are in `docs/research_initiative.md`.


> Primary dev OS: Ubuntu 22.04 (Python 3.12.3); also exercised in Colab (Linux).

### Dataset structure
Images are organised by class folders under `data3a` (extend as needed):
```
data3a/
  training/
    01-minor/
    02-moderate/
    03-severe/
  validation/
    01-minor/
    02-moderate/
    03-severe/
```
Current setup uses three classes (`01-minor`, `02-moderate`, `03-severe`).

## Training Models
All training scripts read the shared hyperparameters from `training_config.py` to ensure fairness. Example commands:
```bash
# Transfer-learning CNN (ResNet18)
python3 scripts/training/train_transfer_cnn.py --data_root data3a --unfreeze

# Transfer-learning CNN (EfficientNet-B0, stronger backbone)
python3 scripts/training/train_efficientnet_b0.py --data_root data3a --strong_aug

# Transfer-learning CNN (MobileNet V3 Small, lightweight)
python3 scripts/training/train_transfer_cnn.py --data_root data3a --unfreeze --model mobilenet_v3_small

# Summarize logged runs
python3 scripts/training/summarize_training_runs.py --output_md training_logs/runs.md
# Optional: generate charts and report
python3 scripts/training/generate_performance_report.py --history training_logs/run_history.csv --charts_dir charts --out_md training_logs/performance_report.md
# Reset logs (start fresh run history)
python3 scripts/training/reset_training_logs.py
```
Each script saves checkpoints under `models/` and writes logs to `training_logs/`.

## Incident Prediction CLI
Run the unified predictor on one or more images:
```bash
python3 incident_predictor.py --model transfer_resnet18 --images path/to/img1.jpg path/to/img2.jpg
```
Models available (exposed): `transfer_efficientnet_b0`, `transfer_resnet18`, `transfer_mobilenet_v3_small`.
(`transfer_efficientnet_b0` is the strongest; train it first if not present.)

## Flask GUI (ICS)
Launch the GUI integrating prediction + routing (uses Part A algorithms):
```bash
FLASK_APP=ics_app.py flask run
# or
python3 ics_app.py
```
Then visit `http://127.0.0.1:5000`, select origin/destination, upload/choose incidents, pick a search algorithm (DFS/BFS/GBFS/A*/IDDFS), and view the top‑k routes plus Leaflet visualization.

- Configure defaults via `ics_config.json` (map file, default model/algorithm, multipliers, max routes).
- Choose the ML model and the number of routes (k). When k>1 the backend uses a time-aware K-shortest routine; k=1 uses the selected search algorithm.
- The map highlights the affected road segment and displays severity, plus start/goal markers and all returned routes.

## Additional Notes
* `ics/routing/search.py` contains the Part A implementations reused by the GUI.
* `training_logger.py` mirrors console logs to files and maintains `training_logs/run_history.csv`.
* `training_config.py` centralises shared hyperparameters; edit this file to run new experiments consistently.
* Archived reference assets (assignment brief, sample images/html) live in `docs/archive/`.
* `ics/routing/k_shortest.py` provides Yen-based k-shortest path computation for reproducible top-k outputs.
* `scripts/run_scenarios.py` executes predefined routing scenarios (see `scenarios/sample_scenarios.json`).
* `scripts/data_prep/preprocess_dataset.py` builds train/val/test splits from a raw, classed image directory (optional resize and deterministic seed).

## Project layout (key modules)
- `ics/web/app.py`: Flask UI + routing/prediction glue (entrypoint shim `ics_app.py`).
- `ics/predict/predictor.py`: all ML model loading/inference (shim `incident_predictor.py` for CLI).
- `ics/routing/`: search algorithms and k-shortest utilities.
- `ics/parser.py`: assignment graph loader used across routing tools.

Refer to `docs/ICS_integration_plan.md` for the integration roadmap and to the assignment report for detailed findings.
