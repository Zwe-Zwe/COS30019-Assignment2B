import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, render_template, request
from PIL import Image

from ics.predict.predictor import IncidentPredictor, PredictionResult, DEFAULT_MULTIPLIERS
from ics.routing.search import dfs, bfs, gbfs, astar, cus1
from ics.parser import parse_assignment_file, Problem
from ics.routing.k_shortest import k_shortest_paths

CONFIG_PATH = Path("ics_config.json")

DEFAULT_CONFIG = {
    "map_file": "heritage_assignment_15_time_asymmetric-1.txt",
    "default_algorithm": "astar",
    "default_model": "transfer_resnet18",
    "max_routes": 5,
    "default_k": 3,
    "severity_multipliers": DEFAULT_MULTIPLIERS,
}


def load_config() -> Dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                cfg.update(data)
        except Exception:
            pass
    return cfg


CONFIG = load_config()
ASSIGNMENT_FILE = Path(CONFIG["map_file"])
MAX_ROUTES = int(CONFIG.get("max_routes", 5))
DEFAULT_K = int(CONFIG.get("default_k", 3))

MODEL_CHOICES = {
    "baseline_cnn": "Baseline CNN",
    "transfer_resnet18": "Transfer ResNet18",
    "transfer_efficientnet_b0": "EfficientNet-B0",
    "transfer_resnet50": "Transfer ResNet50",
    "transfer_vit": "Vision Transformer",
    "resnet_gbt": "ResNet features + GBT",
    "hog_svm": "HOG + SVM",
}

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
predictor_cache: Dict[str, IncidentPredictor] = {}


def get_predictor(model_name: str) -> IncidentPredictor:
    if model_name not in predictor_cache:
        predictor_cache[model_name] = IncidentPredictor(
            model_name=model_name,
            severity_multipliers=CONFIG.get("severity_multipliers", DEFAULT_MULTIPLIERS),
        )
    return predictor_cache[model_name]


def manual_prediction(severity: str) -> PredictionResult:
    multipliers = CONFIG.get("severity_multipliers", DEFAULT_MULTIPLIERS)
    probabilities = {name: 0.0 for name in multipliers}
    probabilities[severity] = 1.0
    class_names = list(multipliers.keys())
    class_id = class_names.index(severity)
    return PredictionResult(
        class_id=class_id,
        class_name=severity,
        probabilities=probabilities,
        multiplier=multipliers[severity],
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


def compute_k_routes(start: str, goal: str, adjustments: Dict[str, float], k: int) -> List[Dict]:
    """Compute k-shortest paths (time-aware) using Yen's algorithm."""
    results = k_shortest_paths(ways, start, goal, k, adjustments=adjustments)
    route_list: List[Dict] = []
    for idx, res in enumerate(results):
        edges = []
        total_cost = 0.0
        for frm, to in zip(res.nodes[:-1], res.nodes[1:]):
            edge = Problem(nodes, ways, start, [goal], adjustments=adjustments, cameras=cameras).get_edge(frm, to)
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
        route_list.append({
            "label": f"K-shortest (route {idx + 1})",
            "algorithm": "k_shortest",
            "nodes": res.nodes,
            "edges": edges,
            "cost": total_cost if total_cost else res.cost,
        })
    return route_list


@app.route("/", methods=["GET", "POST"])
def index():
    prediction: Optional[PredictionResult] = None
    routes: Optional[List[Dict]] = None
    form_errors = []
    route_geometries: List[Dict] = []
    incident_geometry: Optional[Dict] = None

    selected_start = request.form.get("start", meta["start"] or node_choices[0]["id"])
    selected_goal = request.form.get("goal", meta["goals"][0] if meta["goals"] else node_choices[1]["id"])
    selected_way = request.form.get("incident_way", "")
    manual_severity = request.form.get("manual_severity", "")
    selected_algorithm = request.form.get("algorithm", "astar")
    selected_model = request.form.get("model", CONFIG.get("default_model", "transfer_resnet18"))
    requested_k = request.form.get("k_routes", DEFAULT_K)
    if selected_algorithm not in ALGORITHMS:
        selected_algorithm = "astar"
    try:
        k_routes = max(1, min(int(requested_k), MAX_ROUTES))
    except (TypeError, ValueError):
        k_routes = DEFAULT_K

    if request.method == "POST":
        adjustments = {}
        uploaded = request.files.get("incident_image")
        if uploaded and uploaded.filename:
            try:
                data = uploaded.read()
                image = Image.open(io.BytesIO(data))
                predictor = get_predictor(selected_model)
                prediction = predictor.predict(image)
            except Exception as exc:  # pragma: no cover - user input errors
                form_errors.append(f"Failed to process image: {exc}")
        elif selected_way and manual_severity:
            prediction = manual_prediction(manual_severity)

        if selected_way and prediction:
            adjustments[selected_way] = prediction.multiplier

        # Incident geometry for map highlight
        if selected_way:
            match = next((w for w in ways if w["way_id"] == selected_way), None)
            if match:
                incident_geometry = {
                    "from": {
                        "id": match["from"],
                        "lat": nodes[match["from"]]["lat"],
                        "lon": nodes[match["from"]]["lon"],
                        "label": nodes[match["from"]]["label"],
                    },
                    "to": {
                        "id": match["to"],
                        "lat": nodes[match["to"]]["lat"],
                        "lon": nodes[match["to"]]["lon"],
                        "label": nodes[match["to"]]["label"],
                    },
                    "way_id": match["way_id"],
                    "road_name": match["road_name"],
                    "severity": prediction.class_name if prediction else manual_severity,
                }

        if selected_start == selected_goal:
            form_errors.append("Start and destination must be different.")
        else:
            routes = []
            colors = ["#10B981", "#22D3EE", "#F97316", "#9333EA", "#EF4444"]
            if k_routes > 1:
                routes = compute_k_routes(selected_start, selected_goal, adjustments, k_routes)
            else:
                route = compute_route(
                    selected_algorithm,
                    selected_start,
                    selected_goal,
                    adjustments,
                )
                if route:
                    routes = [route]

            if routes:
                for idx, route in enumerate(routes):
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
        manual_severity=manual_severity,
        severity_options=list(CONFIG.get("severity_multipliers", DEFAULT_MULTIPLIERS).keys()),
        model_choices=MODEL_CHOICES,
        selected_model=selected_model,
        k_routes=k_routes,
        max_routes=MAX_ROUTES,
        meta=meta,
        route_geometries=route_geometries,
        incident_geometry=incident_geometry,
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
