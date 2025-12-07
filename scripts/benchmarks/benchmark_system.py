"""
Benchmark routing outcomes across incident severity models.

For a given start/goal and (optional) incident-affected way, this script:
 - loads three predictors (baseline_cnn, transfer_resnet18, hog_svm)
 - runs each predictor on an input image (or manual severity)
 - computes up to 5 unique routes per search algorithm with the predicted multipliers applied

Usage examples:
    python3 benchmark_system.py --image sample.jpg --incident-way 2024 --start 1 --goal 14
    python3 benchmark_system.py --manual-severity 03-severe --incident-way 2024 --algorithms astar gbfs
    python3 benchmark_system.py --image sample.jpg --output training_logs/benchmark_results.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from incident_predictor import IncidentPredictor, PredictionResult, DEFAULT_MULTIPLIERS  # noqa: E402
from ics.routing.search import dfs, bfs, gbfs, astar, cus1  # noqa: E402
from ics.parser import parse_assignment_file, Problem  # noqa: E402

ASSIGNMENT_FILE = Path("heritage_assignment_15_time_asymmetric-1.txt")
DEFAULT_MODELS = ["baseline_cnn", "transfer_resnet18", "hog_svm"]
MAX_ROUTES = 5

ALGORITHMS = {
    "astar": ("A* Search", astar),
    "gbfs": ("Greedy Best-First", gbfs),
    "bfs": ("Breadth-First Search", bfs),
    "dfs": ("Depth-First Search", dfs),
    "cus1": ("Iterative Deepening DFS", cus1),
}


def manual_prediction(severity: str) -> PredictionResult:
    probabilities = {name: 0.0 for name in DEFAULT_MULTIPLIERS}
    probabilities[severity] = 1.0
    class_names = list(DEFAULT_MULTIPLIERS.keys())
    class_id = class_names.index(severity)
    return PredictionResult(
        class_id=class_id,
        class_name=severity,
        probabilities=probabilities,
        multiplier=DEFAULT_MULTIPLIERS[severity],
    )


def compute_route(
    algo_key: str,
    start: str,
    goal: str,
    adjustments: Dict[str, float],
    nodes: Dict,
    ways: List[Dict],
    cameras: Dict,
) -> Optional[Dict]:
    label, algo_fn = ALGORITHMS[algo_key]
    problem = Problem(
        nodes=nodes,
        ways=ways,
        origin=start,
        destinations=[goal],
        adjustments=adjustments,
        cameras=cameras,
    )
    result = algo_fn(problem)
    if not result:
        return None
    _, _, path_str = result
    node_sequence = path_str.split()
    if len(node_sequence) < 2:
        return None

    edges = []
    total_cost = 0.0
    for frm, to in zip(node_sequence[:-1], node_sequence[1:]):
        edge = problem.get_edge(frm, to)
        if edge is None:
            continue
        multiplier = adjustments.get(edge["way_id"], 1.0)
        applied_time = edge["time_min"] * multiplier
        total_cost += applied_time
        edges.append({
            "from_node": frm,
            "to": to,
            "name": edge["road_name"],
            "way_id": edge["way_id"],
            "highway_type": edge["highway_type"],
            "is_camera": edge.get("is_camera", False),
            "base_time": edge["time_min"],
            "applied_time": applied_time,
            "multiplier": multiplier,
        })

    return {
        "label": label,
        "algorithm": algo_key,
        "nodes": node_sequence,
        "edges": edges,
        "cost": total_cost,
    }


def compute_routes_unique(
    algo_key: str,
    start: str,
    goal: str,
    adjustments: Dict[str, float],
    nodes: Dict,
    ways: List[Dict],
    cameras: Dict,
    max_routes: int = MAX_ROUTES,
) -> List[Dict]:
    routes: List[Dict] = []
    working_adjustments = dict(adjustments)
    seen_paths = set()
    for _ in range(max_routes):
        route = compute_route(algo_key, start, goal, working_adjustments, nodes, ways, cameras)
        if not route:
            break
        path_signature = tuple(route["nodes"])
        if path_signature in seen_paths:
            break
        seen_paths.add(path_signature)
        routes.append(route)

        # Penalize used edges to encourage alternative paths
        for edge in route["edges"]:
            working_adjustments[edge["way_id"]] = working_adjustments.get(edge["way_id"], 1.0) + 0.25
    return routes


def run_benchmark(
    models: Sequence[str],
    algorithms: Sequence[str],
    start: str,
    goal: str,
    incident_way: Optional[str],
    image: Optional[Path],
    manual_severity: Optional[str],
    max_routes: int = MAX_ROUTES,
):
    nodes, ways, cameras, meta = parse_assignment_file(ASSIGNMENT_FILE)
    if start is None:
        start = meta["start"] or list(nodes.keys())[0]
    if goal is None:
        goal = (meta["goals"][0] if meta["goals"] else list(nodes.keys())[1])

    predictors = {name: IncidentPredictor(name) for name in models}

    console_lines: List[str] = []
    markdown_lines: List[str] = []

    header = f"Benchmark routes from {start} -> {goal} (incident_way={incident_way or 'none'})"
    console_lines.append(header)
    markdown_lines.append(f"## {header}")
    markdown_lines.append("")

    # Baseline with no incident for comparison
    base_routes = {}
    for algo in algorithms:
        base_routes[algo] = compute_routes_unique(
            algo, start, goal, {}, nodes, ways, cameras, max_routes=1
        )

    for model_name, predictor in predictors.items():
        markdown_lines.append(f"### Model: `{model_name}`")
        console_lines.append(f"\nModel: {model_name}")

        if incident_way:
            if manual_severity:
                prediction = manual_prediction(manual_severity)
            elif image:
                prediction = predictor.predict(image)
            else:
                prediction = None
        else:
            prediction = None

        if prediction:
            console_lines.append(
                f"  Predicted severity={prediction.class_name}, multiplier={prediction.multiplier}"
            )
            markdown_lines.append(
                f"- Prediction: **{prediction.class_name}** (×{prediction.multiplier:.2f})"
            )
            markdown_lines.append("")
            adjustments = {incident_way: prediction.multiplier}
        else:
            console_lines.append("  No incident applied (no image/manual severity or no way provided).")
            markdown_lines.append("- No incident applied (missing image/severity or incident way).")
            adjustments = {}

        for algo in algorithms:
            algo_label = ALGORITHMS[algo][0]
            routes = compute_routes_unique(
                algo, start, goal, adjustments, nodes, ways, cameras, max_routes=max_routes
            )
            if not routes:
                console_lines.append(f"    {algo_label}: no route found.")
                markdown_lines.append(f"- **{algo_label}**: _no route found_")
                continue
            console_lines.append(f"    {algo_label}:")
            markdown_lines.append(f"- **{algo_label}**:")
            for idx, route in enumerate(routes, 1):
                line = f"      route {idx}: {route['cost']:.2f} min | nodes: {' -> '.join(route['nodes'])}"
                console_lines.append(line)
                markdown_lines.append(f"  - route {idx}: {route['cost']:.2f} min — {' -> '.join(route['nodes'])}")

        markdown_lines.append("")
        if incident_way:
            markdown_lines.append("  Baseline (no incident) for reference:")
            for algo in algorithms:
                base = base_routes.get(algo, [])
                if not base:
                    continue
                r0 = base[0]
                markdown_lines.append(
                    f"  - **{ALGORITHMS[algo][0]}** baseline: {r0['cost']:.2f} min — {' -> '.join(r0['nodes'])}"
                )
        markdown_lines.append("")

    return console_lines, markdown_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark routing impact across incident models.")
    parser.add_argument("--start", help="Origin node ID (default: meta start)", default=None)
    parser.add_argument("--goal", help="Destination node ID (default: meta goal)", default=None)
    parser.add_argument("--incident-way", help="Way ID affected by the incident", default=None)
    parser.add_argument("--image", type=Path, help="Incident image to score for severity", default=None)
    parser.add_argument(
        "--manual-severity",
        choices=list(DEFAULT_MULTIPLIERS.keys()),
        help="Fallback severity label when no image is provided.",
        default=None,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=DEFAULT_MODELS,
        help="Which models to benchmark.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(ALGORITHMS.keys()),
        default=["astar", "gbfs", "bfs", "dfs"],
        help="Search algorithms to run.",
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=MAX_ROUTES,
        help="Maximum unique routes to report per algorithm (penalizing repeats).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save Markdown results (e.g., training_logs/benchmark_results.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console_lines, md_lines = run_benchmark(
        models=args.models,
        algorithms=args.algorithms,
        start=args.start,
        goal=args.goal,
        incident_way=args.incident_way,
        image=args.image,
        manual_severity=args.manual_severity,
        max_routes=max(1, min(MAX_ROUTES, args.max_routes)),
    )
    print("\n".join(console_lines))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"\nSaved Markdown results to {args.output}")


if __name__ == "__main__":
    main()
