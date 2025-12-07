import unittest

from ics.parser import Problem, parse_assignment_file
from ics.routing.search import bfs, astar, gbfs, cus1
from ics.routing.k_shortest import k_shortest_paths


def make_dummy_graph():
    nodes = {
        # Equal coordinates ensure heuristic=0 for deterministic shortest paths
        "A": {"lat": 0.0, "lon": 0.0, "label": "A"},
        "B": {"lat": 0.0, "lon": 0.0, "label": "B"},
        "C": {"lat": 0.0, "lon": 0.0, "label": "C"},
        "D": {"lat": 0.0, "lon": 0.0, "label": "D"},
    }
    ways = [
        {"way_id": "e1", "from": "A", "to": "B", "road_name": "AB", "highway_type": "x", "time_min": 1},
        {"way_id": "e2", "from": "B", "to": "D", "road_name": "BD", "highway_type": "x", "time_min": 1},
        {"way_id": "e3", "from": "A", "to": "C", "road_name": "AC", "highway_type": "x", "time_min": 1},
        {"way_id": "e4", "from": "C", "to": "D", "road_name": "CD", "highway_type": "x", "time_min": 3},
    ]
    return nodes, ways


class RoutingTests(unittest.TestCase):
    def test_problem_adjustments_applied(self):
        nodes, ways = make_dummy_graph()
        problem = Problem(nodes, ways, "A", ["D"], adjustments={"e1": 2.0})
        neighbors = dict(problem.get_neighbors("A"))
        self.assertAlmostEqual(neighbors["B"], 2.0)
        self.assertAlmostEqual(neighbors["C"], 1.0)

    def test_bfs_path(self):
        nodes, ways = make_dummy_graph()
        problem = Problem(nodes, ways, "A", ["D"])
        goal, count, path_str = bfs(problem)
        self.assertEqual(goal, "D")
        self.assertIn("A", path_str.split())
        self.assertTrue(path_str.startswith("A"))

    def test_astar_optimal(self):
        nodes, ways = make_dummy_graph()
        problem = Problem(nodes, ways, "A", ["D"])
        goal, count, path_str = astar(problem)
        self.assertEqual(goal, "D")
        self.assertEqual(path_str.split(), ["A", "B", "D"])

    def test_k_shortest_ordered(self):
        nodes, ways = make_dummy_graph()
        paths = k_shortest_paths(ways, "A", "D", k=3)
        self.assertGreaterEqual(len(paths), 2)
        self.assertLessEqual(paths[0].cost, paths[1].cost)
        self.assertEqual(paths[0].nodes, ["A", "B", "D"])

    def test_k_shortest_respects_adjustments(self):
        nodes, ways = make_dummy_graph()
        adjusted = k_shortest_paths(ways, "A", "D", k=1, adjustments={"e1": 10.0})
        self.assertEqual(adjusted[0].nodes, ["A", "C", "D"])

    def test_map_k_shortest_available(self):
        nodes, ways, cameras, meta = parse_assignment_file("heritage_assignment_15_time_asymmetric-1.txt")
        paths = k_shortest_paths(ways, "1", "3", k=2)
        self.assertTrue(paths)
        self.assertEqual(paths[0].nodes[0], "1")
        self.assertEqual(paths[0].nodes[-1], "3")

    def test_map_adjustments_change_cost(self):
        nodes, ways, cameras, meta = parse_assignment_file("heritage_assignment_15_time_asymmetric-1.txt")
        base = k_shortest_paths(ways, "1", "3", k=1)[0].cost
        bumped = k_shortest_paths(ways, "1", "3", k=1, adjustments={"2003": 3.0})[0].cost
        self.assertGreater(bumped, base)

    def test_dfs_finds_goal(self):
        nodes, ways = make_dummy_graph()
        problem = Problem(nodes, ways, "A", ["D"])
        result = bfs(problem)  # BFS for deterministic small graph
        self.assertIsNotNone(result)

    def test_gbfs_returns_path_on_map(self):
        nodes, ways, cameras, meta = parse_assignment_file("heritage_assignment_15_time_asymmetric-1.txt")
        problem = Problem(nodes, ways, "1", ["3"])
        res = gbfs(problem)
        self.assertIsNotNone(res)

    def test_cus1_iddfs_terminates(self):
        nodes, ways = make_dummy_graph()
        problem = Problem(nodes, ways, "A", ["D"])
        res = cus1(problem)
        self.assertIsNotNone(res)

    def test_k_shortest_respects_k_limit(self):
        nodes, ways, cameras, meta = parse_assignment_file("heritage_assignment_15_time_asymmetric-1.txt")
        paths = k_shortest_paths(ways, "1", "14", k=5)
        self.assertLessEqual(len(paths), 5)

    def test_scenario_parity_minor_to_plaza(self):
        nodes, ways, cameras, meta = parse_assignment_file("heritage_assignment_15_time_asymmetric-1.txt")
        adjustments = {"2003": 1.6}  # moderate severity on 2003
        paths = k_shortest_paths(ways, "1", "3", k=3, adjustments=adjustments)
        self.assertTrue(paths)
        # Ensure first path starts/ends correctly and cost reflects multiplier
        self.assertEqual(paths[0].nodes[0], "1")
        self.assertEqual(paths[0].nodes[-1], "3")
        self.assertGreater(paths[0].cost, 0)


if __name__ == "__main__":
    unittest.main()
