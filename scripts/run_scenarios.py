"""
Run reproducible routing scenarios without the Flask UI.

Examples:
    python scripts/run_scenarios.py --scenarios scenarios/sample_scenarios.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from incident_predictor import DEFAULT_MULTIPLIERS
from ics.routing.k_shortest import k_shortest_paths
from ics.parser import parse_assignment_file


def load_config(config_path: Path) -> Dict:
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_scenarios(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Scenario file must contain a list of scenarios")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predefined routing scenarios (top-k shortest).")
    parser.add_argument("--scenarios", type=Path, required=True, help="Path to JSON file with scenarios")
    parser.add_argument("--config", type=Path, default=Path("ics_config.json"), help="Config file for map/multipliers")
    args = parser.parse_args()

    config = load_config(args.config)
    severity_map = config.get("severity_multipliers", DEFAULT_MULTIPLIERS)
    map_file = Path(config.get("map_file", "heritage_assignment_15_time_asymmetric-1.txt"))
    nodes, ways, cameras, meta = parse_assignment_file(map_file)

    scenarios = load_scenarios(args.scenarios)
    for scenario in scenarios:
        start = scenario["start"]
        goal = scenario["goal"]
        k = int(scenario.get("k", 3))
        incidents = scenario.get("incidents", [])
        adjustments = {}
        for inc in incidents:
            sev = inc.get("severity")
            way_id = inc.get("way_id")
            if way_id and sev:
                multiplier = severity_map.get(sev, 1.0)
                adjustments[way_id] = multiplier
        print(f"\nScenario: {scenario.get('name', 'unnamed')}")
        print(f"  Start: {start} -> Goal: {goal} | k={k} | incidents={incidents}")
        paths = k_shortest_paths(ways, start, goal, k, adjustments=adjustments)
        if not paths:
            print("  No path found.")
            continue
        for idx, path in enumerate(paths, start=1):
            print(f"  Route {idx}: cost={path.cost:.2f} | nodes={' -> '.join(path.nodes)}")


if __name__ == "__main__":
    main()
