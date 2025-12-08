# Instructions

## System Requirements
To run the Intelligent Incident Classification System (ICS), the following environment is required:
-   Operating System: Linux (Ubuntu/Pop!_OS recommended), macOS, or Windows 10/11.
-   Python: Version 3.8 or higher.
-   Hardware: 4GB RAM minimum. A GPU (NVIDIA) is recommended for training but not required for inference.

## Installation Guide

1.  Create a virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
2.  Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application
The ICS exposes a web-based user interface.

1.  Open your terminal in the project root.
2.  Run the application server:
    ```bash
    python ics/web/app.py
    ```
3.  Open your web browser and navigate to:
    http://127.0.0.1:5000

## User Guide (The Wizard)
The User Interface is designed as a step-by-step "Wizard" to guide the user through the process.

-   Step 1: Map Selection
    -   The system loads heritage_assignment_15_time_asymmetric-1.txt by default.
    -   You can upload a custom map file (.txt or .osm XML format) if desired.
    ![Map Selection](../screenshots/step1_map_selection.png)

-   Step 2: Define Route
    -   Select your Origin (Start Node) and Destination (Goal Node) from the dropdown lists.
    -   Example: Start at "Jalan Datuk Ajibah Abol" -> Goal at "Jalan Masjid".
    ![Route Selection](../screenshots/step2_route_selection.png)

-   Step 3: Incident Detection (The AI Core)
    -   This is where the Machine Learning integration happens.
    -   Option A (Automatic): Upload an image of a road. The system will use the active Deep Learning model (e.g., EfficientNet-B0) to classify the severity (None, Minor, Moderate, or Severe). If "None" is detected, no penalty is applied. Otherwise, a time penalty is added to the road segment.
    -   Option B (Manual): If no image is available, you can manually select a severity level to test the routing logic.
    ![Incident Configuration](../screenshots/step3_incident_configuration.png)

-   Step 4: Algorithm Configuration
    -   Search Algorithm: Choose A* (recommended), BFS, DFS, or K-Shortest Paths.
    -   Model Selection: Choose EfficientNet-B0 (Accuracy) or MobileNet-V3 (Speed).
    -   Routing Options: Enable "Avoid Tolls" or adjust "K" for multiple routes.
    ![Algorithm Configuration](../screenshots/step4_algorithm_config.png)

-   Step 5: Visualisation & Simulation
    -   The system displays the calculated path on an interactive map.
    -   Simulation Player: Click the "Run Animation" button to see an agent drive the route node-by-node. Use the speed controls to analyze the movement.
    ![Result Visualization](../screenshots/step5_results_visualization.png)
