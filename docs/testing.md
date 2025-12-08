# Testing

Testing was conducted in two phases: ML Model Evaluation and System Integration Testing.

## Test Cases (System Integration)
We created 10 codified test scenarios to ensure robustness.

| ID | Scenario | Input | Expected Outcome | Result |
| :--- | :--- | :--- | :--- | :--- |
| TC-001 | Normal Routing | A*, Start Node 1, Goal Node 13 | Optimal path (16.0 min cost). | PASS |
| TC-002 | Minor Accident | Upload "Minor" image | Prediction "01-minor", Edge Cost x1.2. | PASS |
| TC-003 | Severe Avoidance | Place "Severe" accident on main road | A* chooses a detour (longer dist, shorter time). | PASS |
| TC-004 | K-Shortest Paths | K=3 | Returns 3 distinct valid routes. | PASS |
| TC-005 | Model Consensus | Compare ResNet vs EfficientNet | Both predict same severity class. | PASS |
| TC-006 | Manual Override | Manual "Severe" selection | Weights update without image upload. | PASS |
| TC-007 | BFS Steps | Run BFS | Returns path with fewest nodes (hops). | PASS |
| TC-008 | Simulation | Click "Play" | Marker moves node-by-node along line. | PASS |
| TC-009 | Mobile UI | Resize window to 375px bandwidth | UI adapts, no horizontal scroll. | PASS |
| TC-010 | Disconnected Goal | Set unreachable goal | System handles error gracefully. | PASS |

![Screenshot of the Main Interface](../screenshots/main_interface_placeholder.png)
(Note: Please verify the UI screenshot functionality in your local environment)
