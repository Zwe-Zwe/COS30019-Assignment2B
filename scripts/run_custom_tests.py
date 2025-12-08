
import sys
import os
import glob
import csv
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ics.parser import parse_assignment_file, Problem
from ics.routing.search import astar, bfs, dfs, gbfs, cus1, cus2

ALGORITHMS = {
    "A*": astar,
    "BFS": bfs,
    "DFS": dfs,
    "GBFS": gbfs,
    "IDDFS": cus1,
    "Bi-A*": cus2
}

def calculate_path_cost(problem, path_nodes):
    total_cost = 0.0
    if len(path_nodes) < 2:
        return 0.0
    
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        
        # Find edge
        edge = None
        for e in problem.graph.get(u, []):
            if e['to'] == v:
                edge = e
                break
        
        if edge:
             multiplier = problem.adjustments.get(edge["way_id"], 1.0)
             total_cost += edge["time_min"] * multiplier
             
    return total_cost

def run_tests():
    test_files = sorted(glob.glob(str(PROJECT_ROOT / "test_cases" / "tc*.txt")))
    output_dir = PROJECT_ROOT / "test_results"
    output_csv = output_dir / "algorithm_comparison.csv"
    os.makedirs(output_dir, exist_ok=True)
    
    results_list = []

    # Header for Console
    print(f"{'TEST CASE':<25} | {'ALGO':<5} | {'STATUS':<6} | {'EXPAND':<6} | {'COST':<8} | {'PATH_LEN'}")
    print("-" * 80)
    
    for test_path in test_files:
        test_name = Path(test_path).name.replace(".txt", "")
        
        try:
            nodes, ways, cameras, meta = parse_assignment_file(test_path)
            start = meta.get("start")
            goals = meta.get("goals", [])
            
            # Ensure goals is a list
            if isinstance(goals, str):
                goals = [goals]
            elif goals is None:
                goals = []
            
            if not start or not goals:
                print(f"{test_name:<25} | ALL   | SKIP   | -      | -        | Missing START/GOAL")
                continue
                
            problem = Problem(nodes, ways, start, goals)
            
            for algo_name, algo_func in ALGORITHMS.items():
                try:
                    # Run Algorithm
                    result = algo_func(problem)
                    
                    is_disconnected = "disconnected" in test_name
                    
                    if result:
                        goal_node, expanded, path_str = result
                        path_nodes = path_str.split()
                        cost = calculate_path_cost(problem, path_nodes)
                        path_len = len(path_nodes)
                        
                        status = "FAIL" if is_disconnected else "PASS"
                    else:
                        expanded = 0
                        cost = 0.0
                        path_len = 0
                        status = "PASS" if is_disconnected else "FAIL"
                        
                    # Print
                    print(f"{test_name:<25} | {algo_name:<5} | {status:<6} | {expanded:<6} | {cost:<8.2f} | {path_len}")
                    
                    # Record
                    results_list.append({
                        "Test": test_name,
                        "Algorithm": algo_name,
                        "Status": status,
                        "Nodes_Expanded": expanded,
                        "Path_Cost": cost,
                        "Path_Length": path_len
                    })

                except Exception as e:
                    print(f"{test_name:<25} | {algo_name:<5} | ERROR  | -      | -        | {str(e)}")
                    
        except Exception as e:
             print(f"{test_name:<25} | ERROR | -      | -        | {str(e)}")

    # Save to CSV
    if results_list:
        keys = ["Test", "Algorithm", "Status", "Nodes_Expanded", "Path_Cost", "Path_Length"]
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results_list)
        print(f"\nSaved results to {output_csv}")

if __name__ == "__main__":
    run_tests()
