#!/usr/bin/env python3

from collections import deque
import heapq
from math import inf
from src.parser import Problem

class Node:
    """
    A node in the search tree representing a state in the search space.
    """
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state  # The state (node_id in the graph)
        self.parent = parent  # Parent node in the search tree
        self.action = action  # Action that led to this node
        self.path_cost = path_cost  # Cost from start to this node
        self.depth = 0 if parent is None else parent.depth + 1
        
    def __lt__(self, other):
        """
        Less than comparison for priority queue.
        When all else is equal, nodes should be expanded according to the ascending order.
        """
        return self.state < other.state
    
    def get_path(self):
        """
        Return the path from the root to this node as a list of states.
        """
        node = self
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))
    
    def get_path_string(self):
        """
        Return the path as a string in the required format.
        """
        path = self.get_path()
        return " ".join(map(str, path))

def solution(node, problem, node_count):
    """
    Return the solution in the required format:
    goal node, number of nodes, path
    """
    goal_node = node.state
    path = node.get_path_string()
    return goal_node, node_count, path

def _notify(observer, **payload):
    """
    Send payload to observer callback when provided.
    """
    if observer:
        observer({
            **payload,
            "algorithm": payload.get("algorithm"),
        })


def _goal_signature(node):
    """Return a comparable signature for a goal node."""
    return (
        node.path_cost,
        node.depth,
        tuple(node.get_path()),
    )


def _is_better_goal(candidate, current_best):
    """Determine if the candidate goal improves on the current best solution."""
    if current_best is None:
        return True
    return _goal_signature(candidate) < _goal_signature(current_best)

# Uninformed Search Algorithms

def dfs(problem, observer=None, stop_on_first_goal=True):
    """
    DFS with tie-breaking: (1) depth (implicit in LIFO), (2) chronological order, (3) node ID.
    Uses list with tuples (fifo_counter, node_id, node) to ensure proper ordering.
    """
    origin_node = Node(problem.origin)
    frontier = [(0, problem.origin, origin_node)]  # (fifo, node_id, node)
    fifo_counter = 1

    explored = set()
    node_count = 1

    while frontier:
        _, _, node = frontier.pop()  # LIFO (last in, first out)

        # skip if already expanded
        if node.state in explored:
            continue

        # goal?
        if node.state in problem.destinations:
            _notify(observer, algorithm="dfs", action="goal",
                    current=node.state,
                    frontier=[n.state for _, __, n in frontier],
                    explored=sorted(explored),
                    path=node.get_path(),
                    path_cost=node.path_cost,
                    nodes_created=node_count)
            return solution(node, problem, node_count)

        explored.add(node.state)

        neighbors = problem.get_neighbors(node.state)
        # Sort by node_id ascending for deterministic order
        neighbors.sort(key=lambda x: x[0])
        
        # Collect children with (fifo, node_id, node) tuples
        children = []
        for neighbor_state, cost in neighbors:
            if neighbor_state not in explored:
                child = Node(
                    state=neighbor_state,
                    parent=node,
                    action=(node.state, neighbor_state),
                    path_cost=node.path_cost + cost
                )
                children.append((fifo_counter, neighbor_state, child))
                fifo_counter += 1
                node_count += 1
        
        # Reverse order for LIFO: push higher IDs first so lower IDs pop first
        # When tie on chronological, lower node_id wins
        children.sort(key=lambda x: (-x[0], -x[1]))
        frontier.extend(children)

        _notify(observer, algorithm="dfs", action="expand",
                current=node.state,
                frontier=[n.state for _, __, n in frontier],
                explored=sorted(explored),
                path=node.get_path(),
                path_cost=node.path_cost,
                nodes_created=node_count)

    return None


def bfs(problem, observer=None, stop_on_first_goal=True):
    """
    Breadth-First Search with tie-breaking: (1) depth (implicit in FIFO), (2) chronological order, (3) node ID.
    Uses deque with tuples (fifo_counter, node_id, node) to ensure proper ordering.
    """
    origin_node = Node(problem.origin)
    frontier = deque([(0, problem.origin, origin_node)])  # (fifo, node_id, node)
    frontier_set = {problem.origin}
    fifo_counter = 1
    
    explored = set()
    node_count = 1
    
    while frontier:
        # Sort current frontier by (fifo, node_id) to get the earliest with lowest ID
        frontier_list = list(frontier)
        frontier_list.sort(key=lambda x: (x[0], x[1]))
        
        # Find and remove the best item
        best_item = frontier_list[0]
        frontier.remove(best_item)
        _, _, node = best_item
        frontier_set.discard(node.state)
        
        # Check if the node is a goal
        if node.state in problem.destinations:
            _notify(
                observer,
                algorithm="bfs",
                action="goal",
                current=node.state,
                frontier=[n.state for _, __, n in frontier],
                explored=sorted(explored),
                path=node.get_path(),
                path_cost=node.path_cost,
                nodes_created=node_count,
            )
            return solution(node, problem, node_count)

        if node.state in explored:
            continue
        
        explored.add(node.state)
        
        neighbors = problem.get_neighbors(node.state)
        neighbors.sort(key=lambda x: x[0])  # Sort by node ID
        
        for neighbor_state, cost in neighbors:
            if neighbor_state not in explored and neighbor_state not in frontier_set:
                child = Node(
                    state=neighbor_state,
                    parent=node,
                    action=(node.state, neighbor_state),
                    path_cost=node.path_cost + cost
                )
                frontier.append((fifo_counter, neighbor_state, child))
                frontier_set.add(neighbor_state)
                fifo_counter += 1
                node_count += 1
        
        _notify(
            observer,
            algorithm="bfs",
            action="expand",
            current=node.state,
            frontier=[n.state for _, __, n in frontier],
            explored=sorted(explored),
            path=node.get_path(),
            path_cost=node.path_cost,
            nodes_created=node_count,
        )
    
    return None

# Informed Search Algorithms

class PriorityNode(Node):
    """
    A node with a priority for use in priority queues.
    """
    def __init__(self, state, parent=None, action=None, path_cost=0, priority=0):
        super().__init__(state, parent, action, path_cost)
        self.priority = priority
    
    def __lt__(self, other):
        """
        Less than comparison for priority queue.
        When priorities are equal, compare states.
        """
        if self.priority == other.priority:
            return self.state < other.state
        return self.priority < other.priority

def gbfs(problem, observer=None, stop_on_first_goal=True):
    """
    Greedy Best-First Search with tie-breaking: (1) h-value, (2) chronological order (FIFO), (3) node ID.
    """
    def h(node_state):
        return min(problem.get_euclidean_distance(node_state, dest)
                   for dest in problem.destinations)

    origin_node = PriorityNode(problem.origin, priority=h(problem.origin))
    frontier = [(h(problem.origin), 0, problem.origin, origin_node)]  # (h, fifo, node_id, node)
    heapq.heapify(frontier)
    explored, frontier_set = set(), {problem.origin}
    node_count, fifo_counter = 1, 1

    while frontier:
        _, _, _, node = heapq.heappop(frontier)
        frontier_set.remove(node.state)

        if node.state in problem.destinations:
            _notify(
                observer,
                algorithm="gbfs",
                action="goal",
                current=node.state,
                frontier=[n.state for _, __, ___, n in frontier],
                explored=sorted(explored),
                path=node.get_path(),
                path_cost=node.path_cost,
                nodes_created=node_count,
            )
            return solution(node, problem, node_count)

        explored.add(node.state)
        neighbors = problem.get_neighbors(node.state)
        neighbors.sort(key=lambda x: x[0])

        for neighbor_state, cost in neighbors:
            if neighbor_state not in explored and neighbor_state not in frontier_set:
                h_val = h(neighbor_state)
                child = PriorityNode(neighbor_state, node, (node.state, neighbor_state),
                                     node.path_cost + cost, h_val)
                heapq.heappush(frontier, (h_val, fifo_counter, neighbor_state, child))
                fifo_counter += 1
                frontier_set.add(neighbor_state)
                node_count += 1
                
        _notify(
            observer,
            algorithm="gbfs",
            action="expand",
            current=node.state,
            frontier=[n.state for _, __, ___, n in frontier],
            explored=sorted(explored),
            path=node.get_path(),
            path_cost=node.path_cost,
            nodes_created=node_count,
        )
    return None


def astar(problem, observer=None, stop_on_first_goal=True):
    """
    A* with tie-breaking: (1) f-value (g + h), (2) chronological order (FIFO), (3) node ID.
    Reopen-safe via best_g; skips stale pops.
    """

    def h(state):
        return min(problem.get_euclidean_distance(state, d) for d in problem.destinations)

    h0 = h(problem.origin)
    origin_node = PriorityNode(problem.origin, priority=h0)
    frontier = [(h0, 0, problem.origin, origin_node)]  # (f, fifo, node_id, node)
    heapq.heapify(frontier)
    fifo_counter = 1

    best_g = {problem.origin: 0.0}
    explored = set()
    node_count = 1

    while frontier:
        f_cur, _, _, node = heapq.heappop(frontier)
        g_cur = node.path_cost

        # Skip stale entries
        if g_cur > best_g.get(node.state, float("inf")):
            continue

        explored.add(node.state)

        # Goal check
        if node.state in problem.destinations:
            _notify(
                observer,
                algorithm="astar",
                action="goal",
                current=node.state,
                frontier=[n.state for _, __, ___, n in frontier],
                explored=sorted(explored),
                path=node.get_path(),
                path_cost=node.path_cost,
                nodes_created=node_count,
            )
            return solution(node, problem, node_count)

        # Expand neighbors
        neighbors = problem.get_neighbors(node.state)
        neighbors.sort(key=lambda x: x[0])  # Sort by node ID
        
        for neighbor_state, step_cost in neighbors:
            new_g = g_cur + step_cost
            h_val = h(neighbor_state)
            f_new = new_g + h_val

            # Only push if this improves best_g
            if new_g < best_g.get(neighbor_state, float("inf")):
                best_g[neighbor_state] = new_g
                child = PriorityNode(
                    state=neighbor_state,
                    parent=node,
                    action=(node.state, neighbor_state),
                    path_cost=new_g,
                    priority=f_new
                )
                heapq.heappush(frontier, (f_new, fifo_counter, neighbor_state, child))
                fifo_counter += 1
                node_count += 1

        _notify(
            observer,
            algorithm="astar",
            action="expand",
            current=node.state,
            frontier=[n.state for _, __, ___, n in frontier],
            explored=sorted(explored),
            path=node.get_path(),
            path_cost=node.path_cost,
            nodes_created=node_count,
        )

    return None

# Custom Search Algorithms

def cus1(problem, observer=None, stop_on_first_goal=True):
    """
    Custom Search Strategy 1: Iterative Deepening DFS (IDDFS).
    Tie-breaking: (1) depth limit (implicit), (2) chronological order, (3) node ID.
    """
    node_count = 1
    depth_limit = 0
    
    while True:
        visited = set()
        stack = [(0, problem.origin, Node(problem.origin), 0)]  # (fifo, node_id, node, depth)
        fifo_counter = 1
        
        while stack:
            # Sort by (depth, fifo, node_id) and pop last (LIFO)
            stack.sort(key=lambda x: (x[3], x[0], x[1]))
            _, _, node, depth = stack.pop()
            
            if node.state in problem.destinations:
                _notify(
                    observer,
                    algorithm="cus1",
                    action="goal",
                    current=node.state,
                    frontier=[n.state for _, __, n, ___ in stack],
                    explored=sorted(visited),
                    path=node.get_path(),
                    path_cost=node.path_cost,
                    depth=depth,
                    depth_limit=depth_limit,
                    nodes_created=node_count,
                )
                return solution(node, problem, node_count)
            
            if depth < depth_limit:
                visited.add(node.state)
                
                neighbors = problem.get_neighbors(node.state)
                neighbors.sort(key=lambda x: x[0])
                
                # Collect children
                children = []
                for neighbor_state, cost in neighbors:
                    if neighbor_state not in visited:
                        child = Node(
                            state=neighbor_state,
                            parent=node,
                            action=(node.state, neighbor_state),
                            path_cost=node.path_cost + cost
                        )
                        children.append((fifo_counter, neighbor_state, child, depth + 1))
                        fifo_counter += 1
                        node_count += 1
                
                # Reverse for LIFO: higher fifo and node_id pushed first
                children.sort(key=lambda x: (-x[0], -x[1]))
                stack.extend(children)
                
                _notify(
                    observer,
                    algorithm="cus1",
                    action="expand",
                    current=node.state,
                    frontier=[n.state for _, __, n, ___ in stack],
                    explored=sorted(visited),
                    path=node.get_path(),
                    path_cost=node.path_cost,
                    depth=depth,
                    depth_limit=depth_limit,
                    nodes_created=node_count,
                )

        depth_limit += 1
        
        if depth_limit > 1000:
            return None

def cus2(problem, observer=None):
    """
    CUS2 — Custom Bidirectional A* Search

    Searches forward from origin and backward from all goals simultaneously.
    Tie-breaking: (1) f-value, (2) chronological order (FIFO), (3) node ID (ascending).
    Reopen-safe via best_g_f / best_g_b; skips stale pops.

    Assumptions / fallbacks for reverse expansion:
      - If problem.get_predecessors(state) exists, uses it for the backward side.
      - Else if problem.edges exists (dict or iterable of (u,v)->cost / (u,v,c)), builds reverse adjacency once.
      - Else assumes the graph is undirected and reuses get_neighbors for backward side.

    Returns:
        solution(node, problem, node_count)
        or None if no path exists.
    """

    # -------------------------------
    # Heuristics for both directions
    # -------------------------------
    def h_forward(s):
        # min Euclidean distance to any destination
        return min(problem.get_euclidean_distance(s, d) for d in problem.destinations)

    def h_backward(s):
        # estimated cost from s to origin (used by backward search)
        return problem.get_euclidean_distance(s, problem.origin)

    # -----------------------------------------
    # Build reverse neighbor accessor (once)
    # -----------------------------------------
    get_pred = getattr(problem, "get_predecessors", None)
    reverse_adj = None

    def backward_neighbors(u):
        """Yield (pred, cost) for edges pred -> u."""
        if callable(get_pred):
            return get_pred(u)

        nonlocal reverse_adj
        if reverse_adj is None:
            edges = getattr(problem, "edges", None)
            rev = {}
            if isinstance(edges, dict):
                for (a, b), w in edges.items():
                    rev.setdefault(b, []).append((a, w))
                reverse_adj = rev
            elif edges is not None:
                try:
                    for e in edges:
                        if len(e) == 3:
                            a, b, w = e
                            rev.setdefault(b, []).append((a, w))
                    reverse_adj = rev
                except Exception:
                    reverse_adj = {}
            else:
                reverse_adj = {}

        if reverse_adj:
            return reverse_adj.get(u, [])
        return [(v, w) for (v, w) in problem.get_neighbors(u)]

    # -----------------------------------------------------------
    # Frontier entries are (f, fifo, node_id, PriorityNode)
    # Tie-breaking: (1) f-value, (2) chronological order, (3) node ID
    # -----------------------------------------------------------
    fifo_f = 0
    fifo_b = 0

    # --- forward init (from origin) ---
    g0 = 0.0
    h0 = h_forward(problem.origin)
    start_node = PriorityNode(problem.origin, priority=g0 + h0)
    front_f = [(g0 + h0, fifo_f, problem.origin, start_node)]
    heapq.heapify(front_f)
    fifo_f += 1

    best_g_f = {problem.origin: g0}
    nodes_f = {problem.origin: start_node}
    explored_f = set()

    # --- backward init (from all goals) ---
    front_b = []
    best_g_b = {}
    nodes_b = {}
    explored_b = set()

    for goal in problem.destinations:
        g_back = 0.0
        hb = h_backward(goal)
        node_b = PriorityNode(goal, priority=g_back + hb)
        heapq.heappush(front_b, (g_back + hb, fifo_b, goal, node_b))
        fifo_b += 1
        best_g_b[goal] = 0.0
        nodes_b[goal] = node_b

    node_count = 1 + len(problem.destinations)

    # ---------------------------------
    # Best meeting found so far
    # ---------------------------------
    best_total = inf
    meet_state = None
    meet_f_node = None
    meet_b_node = None

    # Helpers
    def push_children_forward(from_node, g_cur):
        expanded = []
        for v, w in problem.get_neighbors(from_node.state):
            new_g = g_cur + w
            if new_g < best_g_f.get(v, inf):
                hv = h_forward(v)
                fv = new_g + hv
                child = PriorityNode(
                    state=v,
                    parent=from_node,
                    action=(from_node.state, v),
                    path_cost=new_g,
                    priority=fv
                )
                expanded.append((fv, v, child))
        expanded.sort(key=lambda x: (x[0], x[1]))
        return expanded

    def push_children_backward(from_node, g_cur):
        expanded = []
        for p, w in backward_neighbors(from_node.state):
            new_g = g_cur + w
            if new_g < best_g_b.get(p, inf):
                hp = h_backward(p)
                fp = new_g + hp
                child = PriorityNode(
                    state=p,
                    parent=from_node,
                    action=(p, from_node.state),
                    path_cost=new_g,
                    priority=fp
                )
                expanded.append((fp, p, child))
        expanded.sort(key=lambda x: (x[0], x[1]))
        return expanded

    def try_update_best_meeting(s):
        nonlocal best_total, meet_state, meet_f_node, meet_b_node
        if s in best_g_f and s in best_g_b:
            cand = best_g_f[s] + best_g_b[s]
            if cand < best_total:
                best_total = cand
                meet_state = s
                meet_f_node = nodes_f[s]
                meet_b_node = nodes_b[s]

    def top_f(front):
        return front[0][0] if front else inf

    # -------------------------------
    # Main loop
    # -------------------------------
    while front_f or front_b:
        if best_total <= (top_f(front_f) + top_f(front_b)):
            break

        expand_forward = top_f(front_f) <= top_f(front_b)

        if expand_forward:
            if not front_f:
                break
            f_cur, _, node_id, node = heapq.heappop(front_f)
            g_cur = best_g_f.get(node.state, inf)
            if g_cur == inf:
                continue

            explored_f.add(node.state)
            try_update_best_meeting(node.state)

            for fv, v, child in push_children_forward(node, g_cur):
                if child.path_cost < best_g_f.get(v, inf):
                    best_g_f[v] = child.path_cost
                    nodes_f[v] = child
                    heapq.heappush(front_f, (fv, fifo_f, v, child))
                    fifo_f += 1
                    node_count += 1

            _notify(
                observer,
                algorithm="cus2",
                action="expand_f",
                current=node.state,
                frontier=[n.state for _, __, ___, n in front_f],
                explored=sorted(explored_f),
                path=node.get_path(),
                path_cost=node.path_cost,
                nodes_created=node_count,
            )

        else:
            if not front_b:
                break
            f_cur, _, node_id, node = heapq.heappop(front_b)
            g_cur = best_g_b.get(node.state, inf)
            if g_cur == inf:
                continue

            explored_b.add(node.state)
            try_update_best_meeting(node.state)

            for fbv, p, child in push_children_backward(node, g_cur):
                if child.path_cost < best_g_b.get(p, inf):
                    best_g_b[p] = child.path_cost
                    nodes_b[p] = child
                    heapq.heappush(front_b, (fbv, fifo_b, p, child))
                    fifo_b += 1
                    node_count += 1

            _notify(
                observer,
                algorithm="cus2",
                action="expand_b",
                current=node.state,
                frontier=[n.state for _, __, ___, n in front_b],
                explored=sorted(explored_b),
                path=node.get_path(),
                path_cost=node.path_cost,
                nodes_created=node_count,
            )

    if not (meet_state and best_total < inf):
        return None

    # ----------------------------------------
    # Reconstruct path (forward + backward)
    # ----------------------------------------
    f_chain = []
    n = meet_f_node
    while n is not None:
        f_chain.append(n)
        n = n.parent
    f_chain.reverse()

    b_chain_states = []
    b_node = meet_b_node
    first = True
    while b_node is not None:
        if not first:
            b_chain_states.append(b_node.state)
        first = False
        b_node = b_node.parent

    stitched_head = PriorityNode(f_chain[0].state, priority=0.0)
    stitched_head.path_cost = 0.0
    cur = stitched_head

    for nxt in f_chain[1:]:
        step_cost = next(w for (v, w) in problem.get_neighbors(cur.state) if v == nxt.state)
        child = PriorityNode(
            state=nxt.state,
            parent=cur,
            action=(cur.state, nxt.state),
            path_cost=cur.path_cost + step_cost,
            priority=0.0
        )
        cur = child

    for nxt_state in b_chain_states:
        step = None
        for (v, w) in problem.get_neighbors(cur.state):
            if v == nxt_state:
                step = w
                break
        if step is None:
            preds = list(backward_neighbors(cur.state))
            for (p, w) in preds:
                if p == nxt_state:
                    step = w
                    break
        if step is None:
            return {
                "path": [n.state for n in f_chain] + b_chain_states,
                "path_cost": best_total,
                "nodes_created": node_count,
                "meeting": meet_state,
            }

        child = PriorityNode(
            state=nxt_state,
            parent=cur,
            action=(cur.state, nxt_state),
            path_cost=cur.path_cost + step,
            priority=0.0
        )
        cur = child

    goal_node = cur

    _notify(
        observer,
        algorithm="cus2",
        action="goal",
        current=goal_node.state,
        frontier=[n.state for _, __, ___, n in (front_f + front_b)],
        explored=sorted(explored_f | explored_b),
        path=goal_node.get_path(),
        path_cost=goal_node.path_cost,
        nodes_created=node_count,
    )

    return solution(goal_node, problem, node_count)