import io
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from ics.predict.predictor import IncidentPredictor, PredictionResult, DEFAULT_MULTIPLIERS
from ics.routing.search import dfs, bfs, gbfs, astar, cus1
from ics.routing.k_shortest import k_shortest_paths
from ics.parser import parse_assignment_file, Problem

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "ics_config.json"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DEFAULT_CONFIG = {
    "map_file": "heritage_assignment_15_time_asymmetric-1.txt",
    "default_algorithm": "astar",
    "default_model": "transfer_efficientnet_b0",
    "max_routes": 5,
    "default_k": 3,
    "severity_multipliers": DEFAULT_MULTIPLIERS,
}

MODEL_CHOICES = {
    "transfer_efficientnet_b0": "EfficientNet-B0",
    "transfer_resnet18": "ResNet18 (transfer)",
    "transfer_mobilenet_v3_small": "MobileNet V3 Small",
}

ALGORITHMS = {
    "astar": ("A* Search", astar),
    "gbfs": ("Greedy Best-First", gbfs),
    "bfs": ("Breadth-First Search", bfs),
    "dfs": ("Depth-First Search", dfs),
    "cus1": ("Iterative Deepening DFS", cus1),
}

def load_config() -> Dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


CONFIG = load_config()
DEFAULT_MAP_FILE = CONFIG["map_file"]
MAX_ROUTES = int(CONFIG.get("max_routes", 5))
DEFAULT_K = int(CONFIG.get("default_k", 1))


def get_way_display(way):
    return f"{way['way_id']} · {way['road_name']} ({way['from']}→{way['to']})"


app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = "super_secret_key_for_session" # Needed for session access

# Global Cache for parsed maps to avoid re-parsing on every step 2-5 interaction
# Key: filename, Value: (nodes, ways, cameras, meta)
MAP_CACHE = {}

def get_map_data(filename: str):
    if filename not in MAP_CACHE:
        filepath = UPLOAD_DIR / filename if (UPLOAD_DIR / filename).exists() else BASE_DIR / filename
        if not filepath.exists():
             # Fallback to default if file lost
             filepath = BASE_DIR / DEFAULT_MAP_FILE
        
        nodes, ways, cameras, meta = parse_assignment_file(filepath)
        node_choices = sorted(
            [{"id": nid, "label": info["label"]} for nid, info in nodes.items()],
            key=lambda x: int(x["id"]) if x["id"].isdigit() else x["id"]
        )
        way_choices = sorted(
            [{"id": w["way_id"], "label": get_way_display(w), "has_camera": w["way_id"] in cameras} for w in ways],
            key=lambda x: x["label"]
        )
        MAP_CACHE[filename] = {
            "nodes": nodes,
            "ways": ways,
            "cameras": cameras,
            "meta": meta,
            "node_choices": node_choices,
            "way_choices": way_choices
        }
    return MAP_CACHE[filename]

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


def compute_route(algo_key: str, start: str, goal: str, adjustments: Dict[str, float], map_data):
    nodes = map_data["nodes"]
    ways = map_data["ways"]
    cameras = map_data["cameras"]

    label, algo_fn = ALGORITHMS[algo_key]
    
    search_history = []
    def observer(payload):
        if payload.get("action") in ["expand", "goal"]:
             search_history.append({
                 "node": payload.get("current"),
                 "action": payload.get("action"),
                 # Optional: capture frontier/explored if we want detailed viz later
             })

    problem = Problem(
        nodes=nodes,
        ways=ways,
        origin=start,
        destinations=[goal],
        adjustments=adjustments,
        cameras=cameras,
    )
    
    # Pass observer to algo
    result = algo_fn(problem, observer=observer)
    
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
        "history": search_history, # Pass history to frontend
        "edges": edges,
        "cost": total_cost,
    }


def compute_k_routes(algo_key: str, start: str, goal: str, adjustments: Dict[str, float], k: int, map_data) -> List[Dict]:
    nodes = map_data["nodes"]
    ways = map_data["ways"]
    cameras = map_data["cameras"]

    # Hybrid Logic: Run the selected algorithm JUST to capture history for visualization
    search_history = []
    def observer(payload):
        if payload.get("action") in ["expand", "goal", "expand_f", "expand_b"]: # Expand f/b for bidirectional
             search_history.append({
                 "node": payload.get("current"),
                 "action": payload.get("action"),
             })

    _, algo_fn = ALGORITHMS[algo_key]
    problem_for_history = Problem(
        nodes=nodes,
        ways=ways,
        origin=start,
        destinations=[goal],
        adjustments=adjustments,
        cameras=cameras,
    )
    # Run algo to fill search_history (ignore result path)
    algo_fn(problem_for_history, observer=observer)

    # Now run actual K-Shortest logic
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
        
        # Attach history ONLY to the first route (Route 1) which typically matches the optimal A* path
        route_data = {
            "label": f"Route {idx + 1}",
            "algorithm": "k_shortest",
            "nodes": res.nodes,
            "edges": edges,
            "cost": total_cost if total_cost else res.cost,
        }
        if idx == 0:
            route_data["history"] = search_history
        
        route_list.append(route_data)
    return route_list

@app.route("/", methods=["GET", "POST"])
def index():
    # State 1: Determine active step
    # 1 = Map Selection, 2 = Parameters (Wizard entry), 5 = Results
    active_step = 1 
    
    current_map_file = session.get("current_map_file", DEFAULT_MAP_FILE)

    prediction: Optional[PredictionResult] = None
    routes: Optional[List[Dict]] = None
    form_errors = []
    route_geometries: List[Dict] = []
    incident_geometry: Optional[Dict] = None

    # Default Params
    selected_start = ""
    selected_goal = "" 
    selected_way = ""
    manual_severity = ""
    selected_algorithm = CONFIG.get("default_algorithm", "astar")
    selected_model = CONFIG.get("default_model", "transfer_efficientnet_b0")
    k_routes = DEFAULT_K

    if request.method == "POST":
        action = request.form.get("action", "")

        # ACTION: RESET
        if action == "reset":
            session.pop("current_map_file", None)
            return redirect(url_for("index"))

        # ACTION: UPLOAD MAP
        if action == "upload_map":
            file = request.files.get("custom_map")
            if file and file.filename:
                # Security check skipped for assignment context (assuming trusted user)
                filename = file.filename
                save_path = UPLOAD_DIR / filename
                file.save(str(save_path))
                session["current_map_file"] = filename
                current_map_file = filename
                # Proceed to parameters
                active_step = 2
            elif request.form.get("use_default") == "true":
                session["current_map_file"] = DEFAULT_MAP_FILE
                current_map_file = DEFAULT_MAP_FILE
                active_step = 2

        # ACTION: CALCULATE (The final step)
        elif action == "calculate":
            # We assume map is already set in session
            active_step = 5 # Show results
            
            selected_start = request.form.get("start")
            selected_goal = request.form.get("goal")
            selected_way = request.form.get("incident_way", "")
            manual_severity = request.form.get("manual_severity", "")
            selected_algorithm = request.form.get("algorithm", "astar")
            selected_model = request.form.get("model", "transfer_efficientnet_b0")
            requested_k = request.form.get("k_routes", DEFAULT_K)
            
            try:
                k_routes = max(1, min(int(requested_k), MAX_ROUTES))
            except (TypeError, ValueError):
                k_routes = DEFAULT_K

            # Load Data
            map_data = get_map_data(current_map_file)
            nodes = map_data["nodes"]
            ways = map_data["ways"]
            
            # Prediction Logic
            uploaded = request.files.get("incident_image")
            adjustments = {}
            if uploaded and uploaded.filename:
                try:
                    data = uploaded.read()
                    image = Image.open(io.BytesIO(data))
                    predictor = get_predictor(selected_model)
                    prediction = predictor.predict(image)
                except Exception as exc:
                    form_errors.append(f"Failed to process image: {exc}")
            elif selected_way and manual_severity:
                prediction = manual_prediction(manual_severity)

            if selected_way and prediction:
                adjustments[selected_way] = prediction.multiplier

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
                if k_routes > 1:
                    routes = compute_k_routes(selected_algorithm, selected_start, selected_goal, adjustments, k_routes, map_data)
                else:
                    route = compute_route(selected_algorithm, selected_start, selected_goal, adjustments, map_data)
                    routes = [route] if route else []

                if routes:
                    colors = ["#10B981", "#22D3EE", "#F97316", "#9333EA", "#EF4444"]
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
                            "id": idx, # Pass ID for JS selection
                        })
                else:
                    form_errors.append("No route found for the chosen inputs.")

    # Prepare Context
    map_data = get_map_data(current_map_file)
    nodes = map_data["node_choices"]
    all_nodes_dict = map_data["nodes"]
    ways = map_data["way_choices"]
    meta = map_data["meta"]
    
    # Defaults if not set
    if not selected_start and len(nodes) > 0: selected_start = nodes[0]["id"]
    if not selected_goal and len(nodes) > 1: selected_goal = nodes[1]["id"]

    return render_template(
        "index.html",
        active_step=active_step, # VITAL for Wizard
        current_map_name=current_map_file,
        nodes=nodes,
        all_nodes=all_nodes_dict,
        ways=ways,
        all_ways=map_data["ways"], # Pass complete way data for map rendering
        cameras=map_data["cameras"], # Pass cameras to frontend
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
        algorithm_choices=ALGORITHMS,
        selected_algorithm=selected_algorithm,
        meta=meta,
        route_geometries=route_geometries,
        incident_geometry=incident_geometry,
        map_center={
            "lat": all_nodes_dict[selected_start]["lat"] if selected_start else 0,
            "lon": all_nodes_dict[selected_start]["lon"] if selected_start else 0,
        },
        start_coords=None if not selected_start else {
            "lat": all_nodes_dict[selected_start]["lat"],
            "lon": all_nodes_dict[selected_start]["lon"],
            "label": all_nodes_dict[selected_start]["label"],
            "id": selected_start,
        },
        goal_coords=None if not selected_goal else {
            "lat": all_nodes_dict[selected_goal]["lat"],
            "lon": all_nodes_dict[selected_goal]["lon"],
            "label": all_nodes_dict[selected_goal]["label"],
            "id": selected_goal,
        },
    )

@app.route("/charts/<path:filename>")
def charts(filename: str):
    charts_dir = BASE_DIR / "charts"
    return send_from_directory(charts_dir, filename)


if __name__ == "__main__":
    app.run(debug=True)
