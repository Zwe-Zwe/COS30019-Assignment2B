# ICS Integration Plan

## 1. Prediction Service
- Wrap best-performing model (transfer ResNet18 by default) in a dedicated module, e.g. `incident_predictor.py`.
- Responsibilities:
  - Load model weights and class names.
  - Preprocess input image(s) consistently with training pipeline (resize, normalize).
  - Return severity probabilities, predicted class, and calibrated travel-time multiplier.
  - Support batching and device selection (CPU/GPU).
- Extend to use alternative models (baseline, GBT, HOG+SVM) via configuration to compare behaviour.

## 2. Dynamic Edge Cost Pipeline
- Convert severity predictions into travel time adjustments using `ACCIDENT_MULTIPLIER` and/or severity-specific factors.
- Update edge weights inside the routing graph before running path search.
- Allow multiple concurrent incidents: maintain a mapping `way_id -> multiplier`.
- Persist scenario definitions (JSON/YAML) describing which edges are impacted for reproducible testing.

## 3. Route Computation Enhancements
- Implement K-shortest paths (up to 5 routes) between any origin/destination node pair.
  - Reuse or extend Part A search (e.g., Yen’s algorithm or repeated Dijkstra with edge penalties).
  - Include accident-aware travel time in path cost.
- Return detailed route metadata (total time, list of way_ids, camera info) for GUI consumption.
- Provide CLI utilities to request routes for scripted tests.

## 4. GUI & Configuration Layer
- Build a lightweight GUI (Flask web app or desktop) that:
  - Lists all landmarks for origin/destination selection.
  - Accepts incident inputs (upload image, select severity, or choose pre-defined scenarios).
  - Displays Folium map with highlighted incidents and the top-k routes, including travel times.
  - Exposes model/parameter toggles (model choice, number of routes, multiplier override).
- Maintain a configuration file for defaults (model paths, data directories, multipliers).

## 5. Testing & Automation
- Design ≥10 end-to-end scenarios combining predictions + routing; script them for repeatability.
- Capture metrics (per-route travel times, prediction confidences) and screenshots for the report.
- Add unit tests for predictor module and integration tests for the routing pipeline.

## 6. Reporting Artifacts
- Export tables/plots summarizing training experiments (use `summarize_training_runs.py`).
- Generate visual evidence (confusion matrices, route visualisations) for the report’s Testing and Insights sections.
