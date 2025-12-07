"""
K-shortest paths utilities (Yen's algorithm with Dijkstra subroutine).

This module is lightweight and independent of Flask so it can be tested easily.
It expects the same `ways` structure produced by `ics.parser.parse_assignment_file`.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PathResult:
    nodes: List[str]
    cost: float


def _edge_cost(way: Dict, adjustments: Dict[str, float]) -> float:
    return way["time_min"] * adjustments.get(way["way_id"], 1.0)


def _dijkstra(
    ways: Sequence[Dict],
    start: str,
    goal: str,
    adjustments: Dict[str, float],
    banned_edges: Optional[set] = None,
    banned_nodes: Optional[set] = None,
) -> Optional[PathResult]:
    """Shortest path using Dijkstra, respecting banned edges/nodes."""
    banned_edges = banned_edges or set()
    banned_nodes = banned_nodes or set()

    adj: Dict[str, List[Tuple[str, Dict]]] = {}
    for way in ways:
        if (way["from"], way["to"]) in banned_edges:
            continue
        adj.setdefault(way["from"], []).append((way["to"], way))

    frontier: List[Tuple[float, str]] = [(0.0, start)]
    best_g: Dict[str, float] = {start: 0.0}
    parents: Dict[str, Tuple[str, Dict]] = {}

    while frontier:
        cost, node = heapq.heappop(frontier)
        if node in banned_nodes:
            continue
        if node == goal:
            # reconstruct path
            path_nodes = [goal]
            cur = goal
            while cur in parents:
                cur, way = parents[cur]
                path_nodes.append(cur)
            path_nodes.reverse()
            return PathResult(path_nodes, cost)

        for nxt, way in adj.get(node, []):
            if nxt in banned_nodes:
                continue
            step_cost = _edge_cost(way, adjustments)
            new_cost = cost + step_cost
            if new_cost < best_g.get(nxt, float("inf")):
                best_g[nxt] = new_cost
                parents[nxt] = (node, way)
                heapq.heappush(frontier, (new_cost, nxt))

    return None


def k_shortest_paths(
    ways: Sequence[Dict],
    start: str,
    goal: str,
    k: int,
    adjustments: Optional[Dict[str, float]] = None,
) -> List[PathResult]:
    """
    Compute up to k shortest loopless paths using Yen's algorithm.

    Returns a list ordered from best to worst (by total cost). May contain fewer
    than k paths if no alternatives exist.
    """
    if k <= 0:
        return []
    adjustments = adjustments or {}

    first = _dijkstra(ways, start, goal, adjustments)
    if first is None:
        return []
    paths: List[PathResult] = [first]
    candidates: List[Tuple[float, int, PathResult]] = []
    counter = 0

    for kth in range(1, k):
        prev_path = paths[kth - 1].nodes
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[: i + 1]

            banned_edges = set()
            banned_nodes = set(root_path[:-1])

            # Remove edges that would create same root for existing paths
            for p in paths:
                if p.nodes[: i + 1] == root_path and len(p.nodes) > i + 1:
                    banned_edges.add((p.nodes[i], p.nodes[i + 1]))

            spur_res = _dijkstra(
                ways,
                spur_node,
                goal,
                adjustments,
                banned_edges=banned_edges,
                banned_nodes=banned_nodes,
            )
            if spur_res is None:
                continue

            # Combine root and spur (avoid duplicating spur_node)
            total_nodes = root_path[:-1] + spur_res.nodes

            # Compute root cost
            root_cost = 0.0
            for frm, to in zip(root_path[:-1], root_path[1:]):
                for way in ways:
                    if way["from"] == frm and way["to"] == to:
                        root_cost += _edge_cost(way, adjustments)
                        break
            total_cost = root_cost + spur_res.cost

            candidate = PathResult(total_nodes, total_cost)
            # Avoid duplicates
            if all(candidate.nodes != p.nodes for _, __, p in candidates) and all(
                candidate.nodes != p.nodes for p in paths
            ):
                heapq.heappush(candidates, (candidate.cost, counter, candidate))
                counter += 1

        if not candidates:
            break
        _, __, best_candidate = heapq.heappop(candidates)
        paths.append(best_candidate)

    return paths
