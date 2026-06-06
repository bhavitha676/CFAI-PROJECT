"""
================================================================
   DISASTER RELIEF RESOURCE PLANNER
   BTech AI Subject — Covers CO1 to CO6
   
   CO1 → State space representation (zones as states)
   CO2 → BFS, DFS, A* search (find best relief route)
   CO3 → CSP backtracking (assign resources with constraints)
   CO4 → Minimax decision making (adversarial resource choice)
   CO5 → Bayesian network (predict zone risk probability)
   CO6 → Integrated pipeline (all combined with trace output)
================================================================
"""

import math
import random

# ================================================================
# CO1 — STATE SPACE REPRESENTATION
# Formulate the disaster scenario as an AI problem:
#   State    = which zone is currently being helped
#   Actions  = move resources to a zone
#   Goal     = all critical zones (severity >= 7) are helped
#   Cost     = distance between zones
# ================================================================

print("=" * 60)
print("  CO1: STATE SPACE REPRESENTATION")
print("=" * 60)

# Each zone is a state node
zones = {
    "Z1": {"name": "North Riverside",  "severity": 9, "population": 45000, "x": 1, "y": 4},
    "Z2": {"name": "East Hillcrest",   "severity": 6, "population": 20000, "x": 4, "y": 4},
    "Z3": {"name": "Central Floodway", "severity": 10,"population": 30000, "x": 2, "y": 2},
    "Z4": {"name": "South Valley",     "severity": 4, "population": 12000, "x": 4, "y": 1},
    "Z5": {"name": "West Marshland",   "severity": 7, "population": 18000, "x": 1, "y": 1},
    "D":  {"name": "Depot (Start)",    "severity": 0, "population": 0,     "x": 0, "y": 3},
}

# Actions = edges (which zones are connected by road)
roads = {
    "D":  [("Z1", 15), ("Z3", 10)],
    "Z1": [("D",  15), ("Z2", 12), ("Z5", 35)],
    "Z2": [("Z1", 12), ("Z3",  8)],
    "Z3": [("Z2",  8), ("D",  10), ("Z5", 22)],
    "Z4": [("Z5", 14)],
    "Z5": [("Z1", 35), ("Z3", 22), ("Z4", 14)],
}

# Goal test: all zones with severity >= 7 must be reached
def is_goal(visited_zones):
    critical = [z for z in zones if zones[z]["severity"] >= 7 and z != "D"]
    return all(z in visited_zones for z in critical)

# Initial state
initial_state = "D"
print(f"\n  Initial State : {initial_state} (Depot)")
print(f"  Goal          : Visit all zones with severity >= 7")
print(f"  Critical zones: {[z for z in zones if zones[z]['severity'] >= 7 and z != 'D']}")
print(f"\n  State Space:")
for z, info in zones.items():
    print(f"    {z}: {info['name']:<22} severity={info['severity']}  "
          f"coords=({info['x']},{info['y']})")


# ================================================================
# CO2 — GRAPH SEARCH ALGORITHMS
# BFS  → shortest path by hops
# DFS  → explore deep first
# A*   → heuristic (Euclidean distance) for optimal path
# ================================================================

print("\n" + "=" * 60)
print("  CO2: BFS / DFS / A* SEARCH")
print("=" * 60)

# ── BFS ──
def bfs(start, goal, graph):
    queue   = [[start]]
    visited = []
    nodes_expanded = 0
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == goal:
            return path, nodes_expanded
        if node not in visited:
            visited.append(node)
            nodes_expanded += 1
            for neighbour, _ in graph.get(node, []):
                queue.append(path + [neighbour])
    return None, nodes_expanded

# ── DFS ──
def dfs(start, goal, graph):
    stack   = [[start]]
    visited = []
    nodes_expanded = 0
    while stack:
        path = stack.pop()        # pop from end (stack)
        node = path[-1]
        if node == goal:
            return path, nodes_expanded
        if node not in visited:
            visited.append(node)
            nodes_expanded += 1
            for neighbour, _ in graph.get(node, []):
                stack.append(path + [neighbour])
    return None, nodes_expanded

# ── A* ──
def heuristic(node, goal):
    """Euclidean distance between two zone coordinates"""
    x1, y1 = zones[node]["x"], zones[node]["y"]
    x2, y2 = zones[goal]["x"],  zones[goal]["y"]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def astar(start, goal, graph):
    # Each entry: (f_cost, g_cost, path)
    open_list = [(heuristic(start, goal), 0, [start])]
    visited   = []
    nodes_expanded = 0
    while open_list:
        # Pick node with lowest f = g + h (simple sort)
        open_list.sort(key=lambda x: x[0])
        f, g, path = open_list.pop(0)
        node = path[-1]
        if node == goal:
            return path, g, nodes_expanded
        if node not in visited:
            visited.append(node)
            nodes_expanded += 1
            for neighbour, cost in graph.get(node, []):
                new_g = g + cost
                new_h = heuristic(neighbour, goal)
                new_f = new_g + new_h
                open_list.append((new_f, new_g, path + [neighbour]))
    return None, float('inf'), nodes_expanded

# Run all three from Depot to Z4
src, dst = "D", "Z4"
print(f"\n  Searching from {src} to {dst} ({zones[dst]['name']}):\n")

bfs_path, bfs_exp = bfs(src, dst, roads)
dfs_path, dfs_exp = dfs(src, dst, roads)
ast_path, ast_cost, ast_exp = astar(src, dst, roads)

print(f"  BFS  → Path: {' → '.join(bfs_path):<30} Hops: {len(bfs_path)-1}  Nodes expanded: {bfs_exp}")
print(f"  DFS  → Path: {' → '.join(dfs_path):<30} Hops: {len(dfs_path)-1}  Nodes expanded: {dfs_exp}")
print(f"  A*   → Path: {' → '.join(ast_path):<30} Cost: {ast_cost} km  Nodes expanded: {ast_exp}")
print(f"\n  Heuristic used: Euclidean distance from coordinates")
print(f"  A* admissible? YES — h(n) never overestimates real road distance")


# ================================================================
# CO3 — CONSTRAINT SATISFACTION PROBLEM (CSP)
# Variables   = zones (Z1..Z5)
# Domain      = how many food packets to assign (0..300)
# Constraints = 
#   (a) total food assigned <= depot stock (1100)
#   (b) critical zones (severity>=7) must get >= 200 packets
#   (c) non-critical zones get <= 150 packets
# Solve using backtracking
# ================================================================

print("\n" + "=" * 60)
print("  CO3: CSP — RESOURCE ASSIGNMENT WITH BACKTRACKING")
print("=" * 60)

zone_list    = ["Z1", "Z2", "Z3", "Z4", "Z5"]
food_stock   = 1100
food_domains = list(range(0, 501, 50))   # 0, 50, 100, ... 500

def is_consistent(assignment):
    """Check all constraints for current partial assignment"""
    total = sum(assignment.values())
    if total > food_stock:
        return False   # constraint (a)
    for z, val in assignment.items():
        severity = zones[z]["severity"]
        if severity >= 7 and val < 200:
            return False   # constraint (b)
        if severity < 7 and val > 150:
            return False   # constraint (c)
    return True

def backtrack(assignment, remaining_zones):
    if not remaining_zones:
        return assignment if is_consistent(assignment) else None

    # MRV heuristic: pick most constrained variable first (highest severity)
    zone = max(remaining_zones, key=lambda z: zones[z]["severity"])
    rest = [z for z in remaining_zones if z != zone]

    for value in food_domains:
        assignment[zone] = value
        if is_consistent(assignment):
            result = backtrack(assignment, rest)
            if result:
                return result
        del assignment[zone]

    return None   # no valid assignment found

solution = backtrack({}, zone_list)
print(f"\n  Depot food stock : {food_stock} packets")
print(f"  Constraints:")
print(f"    (a) Total assigned <= {food_stock}")
print(f"    (b) Severity >= 7  → must get >= 200 packets")
print(f"    (c) Severity <  7  → max 150 packets")
print(f"\n  CSP Solution (via Backtracking + MRV heuristic):")
if solution:
    total_assigned = 0
    for z in zone_list:
        assigned = solution[z]
        total_assigned += assigned
        sev = zones[z]["severity"]
        tag = "⚠ CRITICAL" if sev >= 7 else ""
        print(f"    {z} {zones[z]['name']:<22} sev={sev}  food={assigned:>4} {tag}")
    print(f"\n    Total assigned : {total_assigned}  |  Remaining : {food_stock - total_assigned}")
else:
    print("  No valid assignment found.")


# ================================================================
# CO4 — MINIMAX DECISION MAKING
# Scenario: Two relief agencies (MAX=our agency, MIN=rival)
#           compete to choose which zone gets the limited
#           medical kits first. MAX wants to maximise lives
#           saved; MIN tries to block/minimise.
# Tree depth = 2 (each agent makes one move)
# ================================================================

print("\n" + "=" * 60)
print("  CO4: MINIMAX — ADVERSARIAL RESOURCE DECISION")
print("=" * 60)

# Utility = severity × (population / 10000) — proxy for lives saved
def utility(zone_id):
    z = zones[zone_id]
    return round(z["severity"] * (z["population"] / 10000), 2)

# Game tree: MAX picks a zone, then MIN can block one sub-choice
game_tree = {
    "ROOT": {                           # MAX's turn
        "Z1": {                         # if MAX picks Z1
            "Z3": utility("Z3"),        # MIN blocks Z3
            "Z5": utility("Z5"),        # MIN blocks Z5
        },
        "Z3": {                         # if MAX picks Z3
            "Z1": utility("Z1"),
            "Z5": utility("Z5"),
        },
        "Z5": {                         # if MAX picks Z5
            "Z1": utility("Z1"),
            "Z3": utility("Z3"),
        },
    }
}

def minimax(node, is_max, depth=0, alpha=-999, beta=999):
    """Minimax with alpha-beta pruning"""
    indent = "    " + "  " * depth

    if isinstance(node, float) or isinstance(node, int):
        print(f"{indent}Leaf value = {node}")
        return node

    if is_max:
        best = -999
        best_move = None
        print(f"{indent}MAX node (depth {depth}):")
        for move, child in node.items():
            val = minimax(child, False, depth + 1, alpha, beta)
            if val > best:
                best = val
                best_move = move
            alpha = max(alpha, best)
            if beta <= alpha:
                print(f"{indent}  ✂ Alpha-beta pruning!")
                break
        print(f"{indent}→ Best move: {best_move}  value={best}")
        return best
    else:
        best = 999
        print(f"{indent}MIN node (depth {depth}):")
        for move, child in node.items():
            val = minimax(child, True, depth + 1, alpha, beta)
            if val < best:
                best = val
            beta = min(beta, best)
            if beta <= alpha:
                print(f"{indent}  ✂ Alpha-beta pruning!")
                break
        print(f"{indent}→ MIN chose value={best}")
        return best

print(f"\n  Utilities (lives-saved proxy):")
for z in ["Z1","Z3","Z5"]:
    print(f"    {z} {zones[z]['name']:<22} utility = {utility(z)}")

print(f"\n  Minimax Tree Trace:")
best_val = minimax(game_tree["ROOT"], is_max=True)
print(f"\n  ✅ Minimax decision value = {best_val}")


# ================================================================
# CO5 — BAYESIAN NETWORK (Risk Prediction)
# Simple Bayes: P(High Risk | Heavy Rain, Low Resources)
#
# Prior probabilities (from historical disaster data):
#   P(Rain=Heavy) = 0.4
#   P(Resources=Low) = 0.5
#
# Conditional: P(HighRisk | Rain, Resources)
#   P(HR | Heavy, Low)  = 0.90
#   P(HR | Heavy, High) = 0.60
#   P(HR | Light, Low)  = 0.55
#   P(HR | Light, High) = 0.20
# ================================================================

print("\n" + "=" * 60)
print("  CO5: BAYESIAN NETWORK — ZONE RISK PREDICTION")
print("=" * 60)

# Conditional probability table (CPT)
cpt = {
    ("Heavy", "Low"):  0.90,
    ("Heavy", "High"): 0.60,
    ("Light", "Low"):  0.55,
    ("Light", "High"): 0.20,
}

p_rain_heavy    = 0.4
p_rain_light    = 0.6
p_res_low       = 0.5
p_res_high      = 0.5

def predict_risk(rain, resources):
    """P(HighRisk | rain, resources) using CPT lookup"""
    return cpt[(rain, resources)]

def marginal_risk():
    """P(HighRisk) = sum over all combinations using total probability"""
    total = 0
    for rain, p_rain in [("Heavy", p_rain_heavy), ("Light", p_rain_light)]:
        for res, p_res in [("Low", p_res_low), ("High", p_res_high)]:
            total += predict_risk(rain, res) * p_rain * p_res
    return total

print(f"\n  Conditional Probability Table (CPT):")
print(f"  {'Rain':<10} {'Resources':<12} P(High Risk)")
print(f"  {'-'*35}")
for (rain, res), prob in cpt.items():
    print(f"  {rain:<10} {res:<12} {prob:.2f}")

print(f"\n  Zone-level predictions (current conditions):")
zone_conditions = {
    "Z1": ("Heavy", "Low"),
    "Z2": ("Light", "High"),
    "Z3": ("Heavy", "Low"),
    "Z4": ("Light", "Low"),
    "Z5": ("Heavy", "High"),
}
for z, (rain, res) in zone_conditions.items():
    risk = predict_risk(rain, res)
    bar = "█" * int(risk * 10) + "░" * (10 - int(risk * 10))
    print(f"  {z} {zones[z]['name']:<22} Rain={rain:<6} Res={res:<5} "
          f"P(Risk)={risk:.2f} [{bar}]")

print(f"\n  Marginal P(High Risk) across all zones = {marginal_risk():.3f}")


# ================================================================
# CO6 — INTEGRATED PIPELINE
# Combine CO1–CO5 into one explainable relief plan:
#   Step 1: Represent state space            (CO1)
#   Step 2: Find best route using A*         (CO2)
#   Step 3: Assign resources via CSP         (CO3)
#   Step 4: Pick zone using Minimax          (CO4)
#   Step 5: Predict risk with Bayes          (CO5)
#   Output: Full reasoning trace             (CO6)
# ================================================================

print("\n" + "=" * 60)
print("  CO6: INTEGRATED PIPELINE — FULL REASONING TRACE")
print("=" * 60)

print("\n  ┌─ STEP 1 [CO1]: State Space ──────────────────────┐")
print(f"  │  Initial state : Depot (D)")
print(f"  │  Goal          : Help all zones with severity ≥ 7")
critical_zones = [z for z in zones if zones[z]["severity"] >= 7 and z != "D"]
print(f"  │  Critical zones: {critical_zones}")
print(f"  └───────────────────────────────────────────────────┘")

print("\n  ┌─ STEP 2 [CO2]: A* Route Planning ────────────────┐")
for target in critical_zones:
    path, cost, _ = astar("D", target, roads)
    print(f"  │  D → {target}: {' → '.join(path):<28} ({cost} km)")
print(f"  └───────────────────────────────────────────────────┘")

print("\n  ┌─ STEP 3 [CO3]: CSP Resource Assignment ──────────┐")
if solution:
    for z in critical_zones:
        print(f"  │  {z} {zones[z]['name']:<22} food = {solution[z]} packets")
print(f"  └───────────────────────────────────────────────────┘")

print("\n  ┌─ STEP 4 [CO4]: Minimax Zone Priority ────────────┐")
print(f"  │  Minimax recommends serving zone with")
print(f"  │  utility value = {best_val}  (maximises lives saved)")
best_zone = max(critical_zones, key=utility)
print(f"  │  Best zone     = {best_zone} ({zones[best_zone]['name']})")
print(f"  └───────────────────────────────────────────────────┘")

print("\n  ┌─ STEP 5 [CO5]: Bayesian Risk Assessment ─────────┐")
for z in critical_zones:
    rain, res = zone_conditions[z]
    risk = predict_risk(rain, res)
    level = "HIGH" if risk >= 0.7 else "MEDIUM" if risk >= 0.4 else "LOW"
    print(f"  │  {z} {zones[z]['name']:<22} P(Risk)={risk:.2f}  → {level}")
print(f"  └───────────────────────────────────────────────────┘")

print("\n  ┌─ FINAL PLAN [CO6]: Explainable Output ───────────┐")
print(f"  │")
for z in sorted(critical_zones, key=utility, reverse=True):
    path, cost, _ = astar("D", z, roads)
    food = solution[z] if solution else "N/A"
    rain, res = zone_conditions[z]
    risk = predict_risk(rain, res)
    print(f"  │  ▶ {z} {zones[z]['name']}")
    print(f"  │    Route   : {' → '.join(path)}  ({cost} km)   [A*]")
    print(f"  │    Food    : {food} packets                    [CSP]")
    print(f"  │    Utility : {utility(z)}                          [Minimax]")
    print(f"  │    Risk    : {risk:.2f}  (Rain={rain}, Res={res})   [Bayes]")
    print(f"  │")
print(f"  └───────────────────────────────────────────────────┘")

print("\n" + "=" * 60)
print("  ✅ All 6 COs demonstrated successfully!")
print("=" * 60)