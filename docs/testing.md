TESTING DOCUMENTATION

This document provides a detailed overview of the testing procedures, strategies, and results for the Intelligent Incident Classification System. It is divided into two primary sections: Machine Learning Model Evaluation and System Integration Testing.

SECTION 1: MACHINE LEARNING MODEL EVALUATION

To ensure the reliability of incident detection, we rigorously evaluated three distinct Neural Network architectures: ResNet18, MobileNet-V3-Small, and EfficientNet-B0.

Evaluation Strategy
- Dataset Split: The dataset was divided into Training (80 percent) and Validation (20 percent) sets.
- Metrics: We monitored Top-1 Accuracy, Loss (Cross-Entropy), and Inference Time (latency in milliseconds).
- Augmentation Check: Models were tested against heavily augmented images (rotations, color shifts) to verify robustness.

Results Summary
1. EfficientNet-B0
   - Validation Accuracy: 80.2 percent
   - Strength: Highest accuracy, best for handling edge cases like nighttime or rainy scenes.
   - Trade-off: Slightly slower inference time compared to MobileNet.

2. MobileNet-V3-Small
   - Validation Accuracy: 77.8 percent
   - Strength: Extremely fast inference (less than 15ms on CPU).
   - Use Case: Verified as the optimal choice for the "High Speed" mode in the application.

3. ResNet18
   - Validation Accuracy: 74.2 percent
   - Status: Used as a stable baseline for comparison.

SECTION 2: SYSTEM INTEGRATION TESTING

System testing focused on the correct interaction between the web interface, the routing algorithms, and the incident predictor. We defined 10 specific Test Cases (TC) covering various complexity levels.

Test Case 001: Normal Routing
- Goal: Verify A-Star algorithm on a standard path.
- Input: Start Node 1 (Jalan Datuk Ajibah Abol), Goal Node 13 (Jalan Masjid).
- Condition: No incidents active.
- Expected Result: Computation of the optimal path with a cost of approx 16.0 minutes.
- Actual Result: PASS. Path matched the manual calculation.

Test Case 002: Minor Incident Avoidance
- Goal: Verify dynamic weight updates.
- Input: Upload an image classified as "Minor Accident".
- Action: System detects class "01-minor" and applies a 1.2x penalty to the edge.
- Expected Result: The algorithm should either stick to the current path (if still optimal) or switch if the penalty makes it too expensive.
- Actual Result: PASS. Cost updated correctly in the debug logs.

Test Case 011: Non-Accident (Safe Road)
- Goal: Verify '00-none' class handling.
- Input: Upload an image of a clear road.
- Action: System predicts '00-none' (No Accident).
- Expected Result: Multiplier remains 1.0; no path change.
- Actual Result: PASS. Verified with image 'data3a/training/00-none/1.jpg'.

Test Case 003: Severe Obstacle Avoidance
- Goal: Ensure safety by avoiding severe accidents.
- Input: Manually flag a main arterial road as "Severe".
- Action: Edge weight multiplied by 2.5x (effectively blocked).
- Expected Result: A-Star immediately reroutes to a longer but safer detour.
- Actual Result: PASS.

Test Case 004: K-Shortest Paths
- Goal: Verify Yen's Algorithm implementation.
- Input: Request K=3 paths.
- Expected Result: System returns exactly 3 distinct routes, ordered by cost.
- Actual Result: PASS. Visualizer displayed three distinct colored lines.

Test Case 005: Model Consensus
- Goal: Consistency check.
- Input: Feed the same ambiguous image to both ResNet and EfficientNet.
- Expected Result: Both models should predict the same class (or close confidence scores).
- Actual Result: PASS. Both predicted "Moderate" with greater than 80 percent confidence.

Test Case 006: Manual Override
- Goal: Test fallback mechanisms.
- Input: User manually selects "Severe" without an image.
- Expected Result: System bypasses the ML predictor and updates weights directly.
- Actual Result: PASS.

Test Case 007: BFS Topology Check
- Goal: Verify structural graph traversal.
- Input: Run BFS between two distant nodes.
- Expected Result: Path returned has the fewest number of "hops" (nodes), regardless of distance.
- Actual Result: PASS. Result differed from A-Star (which optimized for distance), confirming correct behavior.

Test Case 008: Simulation Player
- Goal: UI responsiveness.
- Input: Click "Run Animation" on Route 1.
- Expected Result: The marker moves smoothly node-by-node without crashing the browser.
- Actual Result: PASS.

Test Case 009: Mobile Responsiveness
- Goal: UI adaptability.
- Input: Resize browser window to 375px width (mobile viewport).
- Expected Result: Navigation bar collapses into a "hamburger" menu; no horizontal scrolling.
- Actual Result: PASS.

Test Case 010: Disconnected Graph Handling
- Goal: Error handling.
- Input: Select a Start Node and an unreachable Goal Node (simulated disconnect).
- Expected Result: System returns a "No Path Found" error message instead of crashing.
- Actual Result: PASS.

SECTION 3: ALGORITHM BENCHMARKING

We compared the average execution time of our search algorithms over 100 trials on the standard map.

1. A-Star: 12ms (Fastest optimal search)
2. GBFS: 8ms (Fastest overall, but not always optimal)
3. BFS: 15ms (Slower due to exploring all neighbors)
4. DFS: 5ms (Very fast but produced highly inefficient "spaghetti" paths)

CONCLUSION
All critical features passed their respective test cases. The system demonstrates robust error handling and accurate integration of the Machine Learning components into the pathfinding logic.
