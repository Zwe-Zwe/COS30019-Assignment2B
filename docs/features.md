# Features, Bugs, and Missing Functionality

This document provides a comprehensive overview of the features implemented in the Intelligent Incident Classification System (ICS), along with technical specifications, known limitations, and resolved issues.

## Implemented Features

We are proud to report that all core requirements have been met, along with several advanced enhancements that improve the system's robustness and user experience.

### 1. Data Processing & Machine Learning

The core of our incident detection capability lies in a sophisticated machine learning pipeline designed for high accuracy and rapid inference.

#### Automated Data Pipeline
Our `data_prep` module provides a robust foundation for model training.
-   **Preprocessing**: All input images are automatically resized to **224x224** pixels and normalized using standard ImageNet mean and standard deviation values.
-   **Augmentation**: To prevent overfitting and improve generalization, we employ "strong augmentation" techniques, including:
    -   Random Horizontal/Vertical Flips
    -   Random Rotations (+/- degrees)
    -   Color Jittering (Brightness, Contrast)
-   **Efficiency**: The pipeline is optimized to handle batches effectively, ensuring smooth training loops.

[PLACEHOLDER: Diagram of the Data Preprocessing Pipeline]

#### Transfer Learning Engine
We leveraged Transfer Learning to fine-tune state-of-the-art architectures for our specific traffic incident dataset. Three distinct models were trained and integrated:

1.  **EfficientNet-B0 (Recommended)**
    -   **Optimization**: Trained for 45 epochs with a learning rate of 3e-4.
    -   **Strength**: Uses compound scaling to achieve the highest accuracy among our models.
    -   **Use Case**: Ideal for the default production environment where precision is paramount.

2.  **MobileNet-V3-Small**
    -   **Optimization**: Trained for 35 epochs with a slightly aggressive learning rate of 7e-4.
    -   **Strength**: specific architecture optimized for mobile and edge devices, offering the fastest inference speeds.
    -   **Use Case**: Best for scenarios where low latency is critical.

3.  **ResNet18**
    -   **Optimization**: Trained for 40 epochs.
    -   **Strength**: A Deep Residual Network that provides a stable and well-understood baseline performance.

[PLACEHOLDER: Chart comparing Validation Accuracy of ResNet18 vs EfficientNet vs MobileNet]

#### Inference Service (`IncidentPredictor`)
-   **Real-time Prediction**: The `IncidentPredictor` class loads trained `.pth` models and performs inference in milliseconds.
-   **Dynamic Penalties**: Predictions (None, Minor, Moderate, Severe) are immediately translated into edge weight penalties on the graph. "None" results in no penalty (multiplier 1.0), while others increase the "cost" of traversing affected roads.

---

### 2. Integration & Pathfinding

Our routing engine combines static map data with dynamic real-time updates.

#### Dynamic Graph Weighting
-   The system parses `.txt` map files into an adjacency dictionary structure.
-   **Real-time Updates**: When an incident is detected or manually flagged, the specific edge weights are multiplied by a penalty factor (e.g., 1.5x for Minor, 2.0x for Severe), forcing the algorithms to reconsider the optimal path.

#### Search Algorithms Implemented
We provide a suite of algorithms to demonstrate different graph traversal strategies:

-   **A* Search (A-Star)**:
    -   **Type**: Informed Search.
    -   **Heuristic**: Uses Euclidean distance to the goal.
    -   **Performance**: The most efficient option, consistently finding the optimal path by exploring the most promising nodes first.
    
-   **Greedy Best-First Search (GBFS)**:
    -   **Type**: Informed Search.
    -   **Heuristic**: Uses Euclidean distance.
    -   **Behavior**: prioritizes speed over optimality. It moves towards the goal aggressively and may settle for a sub-optimal path, but often computes faster than A*.

-   **Breadth-First Search (BFS)**:
    -   **Type**: Uninformed (Blind) Search.
    -   **Behavior**: Explores the graph layer-by-layer. Guarantees the shortest path in an unweighted graph, but ignores edge costs (like traffic delays).

-   **Depth-First Search (DFS)**:
    -   **Type**: Uninformed (Blind) Search.
    -   **Behavior**: Explores as deep as possible along each branch before backtracking. Not guaranteed to find the shortest path and can be inefficient in large open maps.

-   **K-Shortest Paths (Yen's Algorithm)**:
    -   **Functionality**: Computes the top *K* Loopless paths between the origin and destination.
    -   **Utility**: Essential for modern navigation, providing users with "Alternative Route 1" and "Alternative Route 2" when the primary route is congested.

[PLACEHOLDER: Visual comparison of A* vs BFS search space exploration]

---

### 3. User Interface & Experience

The application features a modern, web-based interface designed as a "Wizard" to guide users seamlessly through the process.

#### The Wizard Workflow
1.  **Map Selection**: Choose between default heritage maps or upload custom OSM/Text files.
2.  **Route Definition**: Interactive dropdowns to select Start and End nodes.
3.  **Incident Simulation**: Upload accident images to trigger the AI classifier or manually set severity to test the system's response.
4.  **Algorithm Config**: Select the search engine and fine-tune parameters (e.g., K value for K-Shortest paths).
5.  **Visualization**: The final output screen.

[PLACEHOLDER: Screenshot of the Wizard Step 3 (Incident Configuration)]

#### Interactive Visualization
-   **Leaflet.js Map**: We replaced legacy static image plotting with a fully interactive, zoomable, and pannable map using OpenStreetMap tiles.
-   **Simulation Player**: A custom JavaScript-based player allows users to "replay" the agent's journey.
    -   **Features**: Play/Pause controls, Speed adjustment slider, and a progress tracker.
    -   **Visuals**: The path is drawn dynamically, and the "car" icon moves node-by-node, providing clear visual feedback on the route taken.

[PLACEHOLDER: Screenshot of the Simulation Player in action]

#### Responsive Design
-   Built with **Bootstrap 5**, the UI adapts to various screen sizes, ensuring usability on desktops, tablets, and even mobile devices.

---

## Missing Features
-   **None**. All core requirements mandated by the assignment specification have been successfully implemented.
-   **Bonus Features**: We have also addressed suggested bonus components, including the implementation of the K-Shortest Paths algorithm and the integration of multiple Deep Learning architectures.

## Known Bugs & Limitations

### 1. Mobile Responsiveness on Ultra-Small Screens
-   **Issue**: On devices with a viewport width smaller than 300px (e.g., very old smartphones), the navigation bar elements may wrap awkwardly or overlap.
-   **Status**: Minor. Detailed testing confirms usability remains intact on all modern devices (iPhone SE and newer).

### 2. Simulation Visualization (Resolved)
-   **Issue**: In early builds, the simulation player for "Route 1" would incorrectly visualize the entire "Search Tree" (all visited nodes) instead of just the final "Path".
-   **Resolution**: Fixed in the final build. The JavaScript player was updated to strictly iterate over the `route.nodes` array, ensuring only the valid path is animated.

---
