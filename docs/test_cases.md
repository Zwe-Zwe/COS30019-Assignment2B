# Test Cases (≥10)

Automated tests (`python3 -m unittest discover tests`) cover these scenarios:

1. Baseline graph adjustments applied (edges respect multipliers).
2. BFS finds goal on dummy graph.
3. A* returns optimal path on dummy graph.
4. K-shortest ordering on dummy graph.
5. K-shortest adjusts route when edge multipliers increase.
6. Map graph: K-shortest returns valid path 1→3.
7. Map graph: adjustments increase path cost.
8. Map graph: GBFS returns a path.
9. Map graph: IDDFS (cus1) terminates with a path.
10. Map graph: K-shortest caps at requested k.
11. Map graph scenario: 1→3 with a moderate slowdown on way 2003 still yields valid paths.

Run: `python3 -m unittest discover tests` (7 existing + 4 added = 11 test functions).
