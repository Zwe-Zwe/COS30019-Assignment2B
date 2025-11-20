# Incident Classification System – COS30019 Assignment 2B

This repository contains both the ML training pipeline and the integrated ICS (Incident Classification System) for the Kuching heritage map. It builds on Part A (search algorithms) and extends it with multiple ML classifiers, a Flask-based GUI, and top‑k routing.

## Environment
* **Primary OS used during development:** Ubuntu 22.04 (Pop!\_OS derivative); also tested in Google Colab (Linux kernel)
* **Python:** 3.12 (tested on 3.12.3)
* Recommended: create a virtual environment
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

## Dependencies
Install the required packages (all scripts share the same stack):
```bash
pip install torch torchvision scikit-learn scikit-image numpy pandas pillow flask folium
```
> Note: PyTorch install command may vary depending on your platform/CUDA; adjust accordingly.

## Training Models
All training scripts read the shared hyperparameters from `training_config.py` to ensure fairness. Example commands:
```bash
# Baseline CNN
python3 train_baseline_cnn.py --data_root data3a

# Transfer-learning CNN (ResNet18)
python3 train_transfer_cnn.py --data_root data3a --unfreeze

# ResNet feature extractor + Gradient Boosted Trees
python3 train_resnet_features_gbt.py --data_root data3a

# HOG descriptors + SVM
python3 train_hog_svm.py --data_root data3a

# Summarize logged runs
python3 summarize_training_runs.py --output_md training_logs/runs.md
```
Each script saves checkpoints under `models/` and writes logs to `training_logs/`.

## Incident Prediction CLI
Run the unified predictor on one or more images:
```bash
python3 incident_predictor.py --model transfer_resnet18 --images path/to/img1.jpg path/to/img2.jpg
```
Models available: `baseline_cnn`, `transfer_resnet18`, `resnet_gbt`, `hog_svm`.

## Flask GUI (ICS)
Launch the GUI integrating prediction + routing (uses Part A algorithms):
```bash
FLASK_APP=ics_app.py flask run
# or
python3 ics_app.py
```
Then visit `http://127.0.0.1:5000`, select origin/destination, upload/choose incidents, pick a search algorithm (DFS/BFS/GBFS/A*/IDDFS), and view the top‑k routes plus Leaflet visualization.

## Additional Notes
* `search_algorithms.py` contains the Part A implementations reused by the GUI.
* `training_logger.py` mirrors console logs to files and maintains `training_logs/run_history.csv`.
* `training_config.py` centralises shared hyperparameters; edit this file to run new experiments consistently.

Refer to `docs/ICS_integration_plan.md` for the integration roadmap and to the assignment report for detailed findings.
