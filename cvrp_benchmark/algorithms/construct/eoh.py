import numpy as np
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray,
                     rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of IDs of unvisited nodes.
        rest_capacity: rest capacity of vehicle
        demands: demands of nodes
        distance_matrix: Distance matrix of nodes.
    Return:
        ID of the next node to visit.
    """
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot

    current_distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    distance_savings = (distance_matrix[current_node, depot] + distance_matrix[depot
        , feasible_nodes]) - current_distances

    demand_ratio = demands[feasible_nodes] / (rest_capacity + 1e-6)

    avg_distances = np.mean(distance_matrix[feasible_nodes][:, feasible_nodes], axis
        =1)
    density_ratio = current_distances / (avg_distances + 1e-6)

    capacity_factor = rest_capacity / (np.max(demands[feasible_nodes]) + 1e-6)
    proximity_factor = np.mean(current_distances) / (np.mean(depot_distances) + 1e-6)

    w1 = max(0.3, 0.7 - capacity_factor * 0.4)
    w2 = min(0.4, 0.2 + proximity_factor * 0.2)
    w3 = 1.0 - w1 - w2

    weights = w1 * distance_savings + w2 * demand_ratio + w3 * (1 / (density_ratio +
        1e-6))
    next_node = feasible_nodes[np.argmax(weights)]
    return next_node