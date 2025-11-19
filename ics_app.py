import io
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, render_template, request
from PIL import Image

from incident_predictor import IncidentPredictor, PredictionResult, DEFAULT_MULTIPLIERS
from search_algorithms import dfs, bfs, gbfs, astar, cus1
from src.parser import parse_assignment_file, Problem

ASSIGNMENT_FILE = Path("heritage_assignment_15_time_asymmetric-1.txt")
MAX_ROUTES = 5

ALGORITHMS = {
    "astar": ("A* Search", astar),
    "gbfs": ("Greedy Best-First", gbfs),
    "bfs": ("Breadth-First Search", bfs),
    "dfs": ("Depth-First Search", dfs),
    "cus1": ("Iterative Deepening DFS", cus1),
}


def get_way_display(way):
    return f"{way['way_id']} - {way['road_name']} ({way['from']}→{way['to']})"


app = Flask(__name__)
nodes, ways, cameras, meta = parse_assignment_file(ASSIGNMENT_FILE)
node_choices = sorted(
    [{"id": nid, "label": info["label"]} for nid, info in nodes.items()],
    key=lambda x: x["label"]
)
way_choices = sorted(
    [{"id": w["way_id"], "label": get_way_display(w)} for w in ways],
    key=lambda x: x["label"]
)
predictor = IncidentPredictor("transfer_resnet18")


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


def compute_route(algo_key: str, start: str, goal: str, adjustments: Dict[str, float]):
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


@app.route("/", methods=["GET", "POST"])
def index():
    prediction: Optional[PredictionResult] = None
    routes: Optional[List[Dict]] = None
    form_errors = []
    route_geometries: List[Dict] = []

    selected_start = request.form.get("start", meta["start"] or node_choices[0]["id"])
    selected_goal = request.form.get("goal", meta["goals"][0] if meta["goals"] else node_choices[1]["id"])
    selected_way = request.form.get("incident_way", "")
    selected_k = int(request.form.get("num_routes", 3))
    selected_k = max(1, min(MAX_ROUTES, selected_k))
    manual_severity = request.form.get("manual_severity", "")
    selected_algorithm = request.form.get("algorithm", "astar")
    if selected_algorithm not in ALGORITHMS:
        selected_algorithm = "astar"

    if request.method == "POST":
        adjustments = {}
        uploaded = request.files.get("incident_image")
        if uploaded and uploaded.filename:
            try:
                data = uploaded.read()
                image = Image.open(io.BytesIO(data))
                prediction = predictor.predict(image)
            except Exception as exc:  # pragma: no cover - user input errors
                form_errors.append(f"Failed to process image: {exc}")
        elif selected_way and manual_severity:
            prediction = manual_prediction(manual_severity)

        if selected_way and prediction:
            adjustments[selected_way] = prediction.multiplier

        if selected_start == selected_goal:
            form_errors.append("Start and destination must be different.")
        else:
            routes = []
            colors = ["#1D4ED8", "#DC2626", "#059669", "#C2410C", "#9333EA"]
            working_adjustments = dict(adjustments)
            for idx in range(selected_k):
                active_adjustments = dict(working_adjustments)
                route = compute_route(
                    selected_algorithm,
                    selected_start,
                    selected_goal,
                    active_adjustments,
                )
                if not route:
                    break
                route["label"] = f"{route['label']} (route {idx+1})"
                routes.append(route)

                coords = []
                for node_id in route["nodes"]:
                    info = nodes[node_id]
                    coords.append({"lat": info["lat"], "lon": info["lon"], "label": info["label"]})
                route_geometries.append({
                    "coords": coords,
                    "color": colors[idx % len(colors)],
                    "label": route["label"],
                    "cost": route["cost"],
                })

                # Penalize used edges for alternative routes
                for edge in route["edges"]:
                    working_adjustments[edge["way_id"]] = working_adjustments.get(edge["way_id"], 1.0) + 0.25

            if not routes:
                form_errors.append("No route found for the chosen inputs.")

    return render_template(
        "index.html",
        nodes=node_choices,
        ways=way_choices,
        prediction=prediction,
        routes=routes,
        errors=form_errors,
        selected_start=selected_start,
        selected_goal=selected_goal,
        selected_way=selected_way,
        selected_k=selected_k,
        manual_severity=manual_severity,
        severity_options=list(DEFAULT_MULTIPLIERS.keys()),
        meta=meta,
        route_geometries=route_geometries,
        map_center={
            "lat": nodes[selected_start]["lat"],
            "lon": nodes[selected_start]["lon"],
        },
        start_coords={
            "lat": nodes[selected_start]["lat"],
            "lon": nodes[selected_start]["lon"],
            "label": nodes[selected_start]["label"],
            "id": selected_start,
        },
        goal_coords={
            "lat": nodes[selected_goal]["lat"],
            "lon": nodes[selected_goal]["lon"],
            "label": nodes[selected_goal]["label"],
            "id": selected_goal,
        },
        algorithm_choices=ALGORITHMS,
        selected_algorithm=selected_algorithm,
    )


if __name__ == "__main__":
    app.run(debug=True)
