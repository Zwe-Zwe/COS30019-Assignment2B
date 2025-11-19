import math
from typing import Dict, List, Tuple


def parse_assignment_file(path):
    section = None
    nodes: Dict[str, Dict] = {}
    ways: List[Dict] = []
    cameras: Dict[str, str] = {}
    meta = {"start": None, "goals": [], "accident_multiplier": None}

    def is_header(line):
        return line.startswith("[") and line.endswith("]")

    def ignore(line):
        return (not line.strip()) or line.strip().startswith("#")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if ignore(line):
                continue
            if is_header(line):
                section = line.upper()
                continue

            if section == "[NODES]":
                node_id, lat, lon, label = [x.strip() for x in line.split(",", 3)]
                nodes[node_id] = {"lat": float(lat), "lon": float(lon), "label": label}
            elif section == "[WAYS]":
                parts = [x.strip() for x in line.split(",", 5)]
                ways.append({
                    "way_id": parts[0],
                    "from": parts[1],
                    "to": parts[2],
                    "road_name": parts[3],
                    "highway_type": parts[4],
                    "time_min": float(parts[5]),
                })
            elif section == "[CAMERAS]":
                way_id, asset = [x.strip() for x in line.split(",", 1)]
                cameras[way_id] = asset
            elif section == "[META]":
                parts = [x.strip() for x in line.split(",")]
                key = parts[0].upper()
                if key == "START":
                    meta["start"] = parts[1]
                elif key == "GOAL":
                    meta["goals"] = parts[1:]
                elif key == "ACCIDENT_MULTIPLIER":
                    meta["accident_multiplier"] = float(parts[1])

    return nodes, ways, cameras, meta


class Problem:
    """
    Search problem representation compatible with Part A algorithms.
    """

    def __init__(self, nodes, ways, origin, destinations,
                 adjustments=None, cameras=None):
        self.nodes = nodes
        self.origin = origin
        self.destinations = destinations
        self.cameras = cameras or {}
        self.graph: Dict[str, List[Dict]] = {}
        self.adjustments = adjustments or {}
        self._ways_source = [dict(w) for w in ways]
        self._build_graph(self._ways_source)

    def _build_graph(self, ways):
        for way in ways:
            entry = dict(way)
            entry["is_camera"] = entry["way_id"] in self.cameras
            self.graph.setdefault(entry["from"], []).append(entry)

    def get_neighbors(self, node_state) -> List[Tuple[str, float]]:
        neighbors = []
        for edge in self.graph.get(node_state, []):
            multiplier = self.adjustments.get(edge["way_id"], 1.0)
            neighbors.append((edge["to"], edge["time_min"] * multiplier))
        return neighbors

    def get_euclidean_distance(self, a_state, b_state) -> float:
        a = self.nodes[a_state]
        b = self.nodes[b_state]
        # Simple haversine distance in km
        lat1, lon1, lat2, lon2 = map(math.radians, [a["lat"], a["lon"], b["lat"], b["lon"]])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 6371 * 2 * math.atan2(math.sqrt(h), math.sqrt(1-h))

    def get_edge(self, from_state, to_state):
        for edge in self.graph.get(from_state, []):
            if edge["to"] == to_state:
                return edge
        return None

    def clone_with_adjustments(self, adjustments):
        merged = dict(self.adjustments)
        merged.update(adjustments)
        return Problem(self.nodes, self._ways_source, self.origin, self.destinations,
                       adjustments=merged, cameras=self.cameras)
