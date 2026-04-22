
import numpy as np
from typing import List, Tuple
from collections import defaultdict

def solve_vrp(distance_matrix: np.ndarray, demands: List[int], capacity: int, depot_index: int = 0) -> List[List[int]]:

    n = len(distance_matrix)
    customers = [i for i in range(n) if i != depot_index and demands[i] > 0]

    routes = [[depot_index, c, depot_index] for c in customers]
    route_demands = [demands[c] for c in customers]

    savings = []
    for i in range(len(customers)):
        for j in range(i+1, len(customers)):
            c1, c2 = customers[i], customers[j]
            s = distance_matrix[depot_index][c1] + distance_matrix[depot_index][c2] - distance_matrix[c1][c2]
            savings.append((s, i, j))

    savings.sort(reverse=True, key=lambda x: x[0])

    route_indices = list(range(len(routes)))
    customer_to_route = {i: i for i in range(len(customers))}
    
    for s, i, j in savings:
        ri = customer_to_route[i]
        rj = customer_to_route[j]
        
        if ri == rj:
            continue
        
        route1 = routes[ri]
        route2 = routes[rj]
        combined_demand = route_demands[ri] + route_demands[rj]
        
        if combined_demand > capacity:
            continue

        merged1 = route1[:-1] + route2[1:]
        cost1 = calculate_route_cost(merged1, distance_matrix)
        
        merged2 = route2[:-1] + route1[1:]
        cost2 = calculate_route_cost(merged2, distance_matrix)
        
        if cost1 < cost2:
            merged = merged1
            cost = cost1
        else:
            merged = merged2
            cost = cost2

        routes[ri] = merged
        route_demands[ri] = combined_demand
        routes[rj] = None
        route_demands[rj] = 0
        
        for c in route2[1:-1]:
            customer_idx = customers.index(c)
            customer_to_route[customer_idx] = ri

    routes = [r for r in routes if r is not None]

    for i in range(len(routes)):
        if len(routes[i]) > 3:
            routes[i] = two_opt_optimization(routes[i], distance_matrix)
    
    return routes

def calculate_route_cost(route: List[int], distance_matrix: np.ndarray) -> float:

    cost = 0
    for i in range(len(route)-1):
        cost += distance_matrix[route[i]][route[i+1]]
    return cost

def two_opt_optimization(route: List[int], distance_matrix: np.ndarray) -> List[int]:

    improved = True
    best_route = route.copy()
    best_cost = calculate_route_cost(route, distance_matrix)
    
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                if j-i == 1:
                    continue

                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_cost = calculate_route_cost(new_route, distance_matrix)
                
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
        
        route = best_route
    
    return best_route