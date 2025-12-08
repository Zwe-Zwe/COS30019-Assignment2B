# Features, Bugs, and Missing Functionality

## Implemented Features
We are proud to report that all requirements have been met, along with several advanced enhancements.

### Data Processing & Machine Learning
-   Automated Data Pipeline: A robust data_prep module that handles image resizing (224x224), normalization (ImageNet stats), and augmentation (random flips/rotations) to prevent overfitting.
-   Transfer Learning Engine: We implemented three distinct architectures:
    -   ResNet18: A residual network providing a stable baseline.
    -   EfficientNet-B0: A compound-scaled network offering the best accuracy.
    -   MobileNet-V3-Small: A lightweight network optimized for speed.
-   Inference Service: A dedicated IncidentPredictor class that loads saved .pth models and runs inference in milliseconds.

### Integration & Pathfinding
-   Dynamic Graph Weighting: The system parses .txt map files into an adjacency dictionary. When an incident is detected, the edge weights are updated in real-time before the search algorithm runs.
-   Algorithms Implemented:
    -   A* Search: Uses Euclidean distance heuristic for optimal routing.
    -   Greedy Best-First Search: Faster but non-optimal.
    -   BFS / DFS: Unweighted blind searches for structural analysis.
    -   K-Shortest Paths (Yen's Algorithm): Returns the top K routes, essential for offering drivers alternatives during heavy traffic.

### User Interface
-   Interactive Map (Leaflet.js): Replaced static images with a zoomable, pannable map using OpenStreetMap tiles.
-   Universal Simulation Player: A custom JavaScript feature allowing users to "play back" the route journey step-by-step.
-   Responsive Design: Built with Bootstrap 5 to ensure usability on tablets and mobile devices.

## Missing Features
-   None. All core requirements and "bonus" suggestion components have been addressed.

## Known Bugs
-   Minor: On very small screens (<300px width), the navigation bar may wrap awkwardly.
-   Resolved: Previously, the simulation player for Route 1 visualized the "Search Tree" instead of the "Path". This was fixed in the final build by forcing the player to read route.nodes.
