import sys
import math
import folium
import xml.etree.ElementTree as ET
import heapq

###############################################################################
# 1. Parse assignment file (6-column WAYS: time in minutes, directed edges)
###############################################################################

def split_csv_allow_commas(line, min_fields):
    parts = []
    buf = []
    depth = 0
    for ch in line:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth = max(depth - 1, 0)
            buf.append(ch)
        elif ch == ',':
            if depth == 0:
                parts.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    if len(parts) < min_fields:
        raise ValueError(f"Line '{line}' parsed into too few fields: {parts}")
    return parts


def parse_assignment_file(path):
    section = None
    nodes = {}
    ways = []
    cameras = {}
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
                p = split_csv_allow_commas(line, 4)
                nid, lat, lon, label = p[0], float(p[1]), float(p[2]), p[3]
                nodes[nid] = {"lat": lat, "lon": lon, "label": label}

            elif section == "[WAYS]":
                # way_id,from_node,to_node,road_name,highway_type,travel_time_min
                p = split_csv_allow_commas(line, 6)
                ways.append({
                    "way_id": p[0],
                    "from": p[1],
                    "to": p[2],
                    "road_name": p[3],
                    "highway_type": p[4],
                    "time_min": float(p[5]),
                })

            elif section == "[CAMERAS]":
                p = split_csv_allow_commas(line, 2)
                cameras[p[0]] = p[1]

            elif section == "[META]":
                p = [x.strip() for x in line.split(",")]
                key = p[0].upper()
                if key == "START":
                    meta["start"] = p[1]
                elif key == "GOAL":
                    meta["goals"] = p[1:]
                elif key == "ACCIDENT_MULTIPLIER":
                    meta["accident_multiplier"] = float(p[1])

    return nodes, ways, cameras, meta

###############################################################################
# 2. Load map.osm into a road graph
###############################################################################

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dla = la2 - la1
    dlo = math.radians(lon2 - lon1)
    a = math.sin(dla/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlo/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def load_osm_graph(osm_path):
    tree = ET.parse(osm_path)
    root = tree.getroot()

    osm_nodes = {}
    for n in root.findall("node"):
        nid = n.attrib["id"]
        lat = float(n.attrib["lat"])
        lon = float(n.attrib["lon"])
        osm_nodes[nid] = (lat, lon)

    graph = {nid: [] for nid in osm_nodes}

    for w in root.findall("way"):
        nd_refs = [nd.attrib["ref"] for nd in w.findall("nd")]
        tags = {t.attrib.get("k"): t.attrib.get("v") for t in w.findall("tag")}
        if "highway" not in tags:
            continue

        for i in range(len(nd_refs) - 1):
            a, b = nd_refs[i], nd_refs[i+1]
            if a in osm_nodes and b in osm_nodes:
                lat1, lon1 = osm_nodes[a]
                lat2, lon2 = osm_nodes[b]
                dist = haversine_km(lat1, lon1, lat2, lon2)
                graph[a].append((b, dist))
                graph[b].append((a, dist))

    # Remove isolated nodes
    isolated = [nid for nid, nbrs in graph.items() if len(nbrs) == 0]
    for nid in isolated:
        del graph[nid]

    return osm_nodes, graph

###############################################################################
# 3. K-nearest snapping
###############################################################################

def k_nearest_graph_nodes(lat, lon, osm_nodes, graph, k=5):
    candidates = []
    for nid, (nlat, nlon) in osm_nodes.items():
        if nid not in graph:
            continue
        d_lat = lat - nlat
        d_lon = lon - nlon
        d2 = d_lat*d_lat + d_lon*d_lon
        candidates.append((d2, nid))
    candidates.sort(key=lambda x: x[0])
    return [nid for d2, nid in candidates[:k]]

###############################################################################
# 4. Dijkstra in road graph
###############################################################################

def dijkstra_path(graph, start_id, goal_id):
    dist = {start_id: 0.0}
    prev = {}
    pq = [(0.0, start_id)]
    visited = set()

    while pq:
        cur_d, cur = heapq.heappop(pq)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal_id:
            break

        for nbr, w in graph.get(cur, []):
            nd = cur_d + w
            if nbr not in dist or nd < dist[nbr]:
                dist[nbr] = nd
                prev[nbr] = cur
                heapq.heappush(pq, (nd, nbr))

    if goal_id not in dist:
        return None

    path = []
    node = goal_id
    while True:
        path.append(node)
        if node == start_id:
            break
        node = prev.get(node)
        if node is None:
            return None
    path.reverse()
    return path


def path_length_km(path, graph):
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        for nbr, w in graph.get(a, []):
            if nbr == b:
                total += w
                break
    return total

###############################################################################
# 5. Folium helpers
###############################################################################

def color_for_highway(hwy_type):
    h = (hwy_type or "").lower()
    if h == "primary":
        return "deepskyblue"
    elif h == "secondary":
        return "purple"
    elif h == "tertiary":
        return "darkblue"
    elif h == "service":
        return "slategray"
    else:
        return "white"


def get_map_center(nodes):
    lats = [info["lat"] for info in nodes.values()]
    lons = [info["lon"] for info in nodes.values()]
    return sum(lats)/len(lats), sum(lons)/len(lons)

###############################################################################
# 6. Main visualiser
###############################################################################

def visualize_with_roads(nodes, ways, cameras, meta,
                         osm_nodes, graph,
                         out_html="heritage_map_roads.html",
                         k_snap=5):

    center_lat, center_lon = get_map_center(nodes)
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=18,
                   tiles="OpenStreetMap")

    # Precompute snap candidates
    snap_candidates = {}
    for nid, info in nodes.items():
        snap_candidates[nid] = k_nearest_graph_nodes(
            info["lat"], info["lon"], osm_nodes, graph, k=k_snap
        )

    print("[DEBUG] Snap candidates (first 3 per node):")
    for nid, cand in snap_candidates.items():
        print(f"  Node {nid} ({nodes[nid]['label']}): {cand[:3]}")

    # Draw each edge following road geometry if possible
    for w in ways:
        way_id = w["way_id"]
        u = w["from"]
        v = w["to"]
        rn = w["road_name"]
        hwy = w["highway_type"]
        time_min = w["time_min"]

        best_path = None
        best_len = None

        for u_osm in snap_candidates[u]:
            for v_osm in snap_candidates[v]:
                path = dijkstra_path(graph, u_osm, v_osm)
                if path is None:
                    continue
                plen = path_length_km(path, graph)
                if best_len is None or plen < best_len:
                    best_len = plen
                    best_path = path

        if best_path is None:
            latlngs = [(nodes[u]["lat"], nodes[u]["lon"]),
                       (nodes[v]["lat"], nodes[v]["lon"])]
        else:
            latlngs = [(osm_nodes[nid][0], osm_nodes[nid][1]) for nid in best_path]

        is_camera = way_id in cameras
        if is_camera:
            line_color = "crimson"
            weight = 6
            dash_array = "8,4"
        else:
            line_color = color_for_highway(hwy)
            weight = 5 if hwy in ["primary", "secondary"] else 4
            dash_array = None

        tooltip_text = f"{rn} ({time_min} min)"
        popup_html = (
            f"<b>{rn}</b><br>"
            f"way_id: {way_id}<br>"
            f"type: {hwy}<br>"
            f"time: {time_min} min<br>"
        )
        if is_camera:
            popup_html += "<b>CAMERA MONITORED</b><br>Accident → time multiplied"

        folium.PolyLine(
            locations=latlngs,
            color=line_color,
            weight=weight,
            dash_array=dash_array,
            tooltip=tooltip_text,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

    # Node markers
    start_node = meta["start"]
    goal_nodes = set(meta["goals"])

    for nid, info in nodes.items():
        lat, lon, label = info["lat"], info["lon"], info["label"]

        if nid == start_node:
            fill_color = "green"
            role = "START"
        elif nid in goal_nodes:
            fill_color = "blue"
            role = "GOAL"
        else:
            fill_color = "white"
            role = "Node"

        folium.CircleMarker(
            location=(lat, lon),
            radius=8,
            color="black",
            weight=1,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>Node {nid}: {label}</b><br>"
                f"lat: {lat:.6f}<br>lon: {lon:.6f}<br>{role}",
                max_width=250
            ),
            tooltip=f"{nid}: {label} ({role})"
        ).add_to(m)

        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size:10px;
                    font-weight:bold;
                    color:black;
                    background-color:rgba(255,255,255,0.8);
                    border:1px solid black;
                    border-radius:3px;
                    padding:2px 3px;
                    white-space:nowrap;
                ">{nid}: {label}</div>
                """
            )
        ).add_to(m)

    # Legend
    acc_mult = meta["accident_multiplier"]
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        background: rgba(255,255,255,0.9);
        padding: 8px 12px;
        border: 1px solid #333;
        border-radius: 4px;
        font-size: 12px;
        line-height: 1.4;
    ">
    <b>Kuching Heritage Graph</b><br>
    <span style="color:green;font-weight:bold;">●</span> START ({start_node})<br>
    <span style="color:blue;font-weight:bold;">●</span> GOAL(s) {', '.join(goal_nodes)}<br>
    <span style="color:crimson;font-weight:bold;">▬ ▬</span> Camera road (accident → time ×{acc_mult})<br>
    <span style="color:deepskyblue;font-weight:bold;">▬</span> Primary road<br>
    <span style="color:gold;font-weight:bold;">▬</span> Secondary road<br>
    <span style="color:darkorange;font-weight:bold;">▬</span> Tertiary road<br>
    <span style="color:slategray;font-weight:bold;">▬</span> Service/local<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(out_html)
    print(f"[OK] Saved map with road-following edges to {out_html}")

###############################################################################
# 7. CLI
###############################################################################

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_assignment_folium_roads_knearest.py heritage_assignment_15_time_asymmetric.txt map.osm")
        sys.exit(1)

    assignment_path = sys.argv[1]
    osm_path = sys.argv[2]

    nodes, ways, cameras, meta = parse_assignment_file(assignment_path)
    osm_nodes, graph = load_osm_graph(osm_path)

    print(f"[INFO] Loaded assignment: {len(nodes)} nodes; {len(ways)} ways")
    print(f"[INFO] Loaded OSM graph: {len(osm_nodes)} nodes; "
          f"{sum(len(v) for v in graph.values())} directed edges")

    visualize_with_roads(nodes, ways, cameras, meta, osm_nodes, graph)

if __name__ == "__main__":
    main()
