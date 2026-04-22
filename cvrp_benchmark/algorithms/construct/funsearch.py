import numpy as np
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: np.ndarray,
                     demands: np.ndarray, distance_matrix: np.ndarray) -> int:
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
    best_score = float('-inf')
    next_node = depot

    # Get the total demand of unvisited nodes to assess future needs
    total_future_demand = sum(demands[unvisited_nodes])

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]

        # Ensure demand can be fulfilled
        if demand <= rest_capacity:
            # Calculate score based on demand and adjusted by the distance
            score = (demand / distance) if distance > 0 else float('inf')
            score -= (distance / (rest_capacity + 1e-5))  # Penalty for distance

            # Include a scale for total future demand, encouraging visiting nodes with higher total demands among unvisited nodes
            future_demand_score = min(total_future_demand - demand, rest_capacity) / (distance + 1e-5)
            score += future_demand_score

            # If remaining capacity is low and we are close to depot, heavily favor returning to depot
            if len(unvisited_nodes) == 1 and rest_capacity < demands[depot]:
                score += 1000  # Arbitrary high score to favor depot return

            # Update the best score and next node if current score exceeds best score
            if score > best_score:
                best_score = score
                next_node = node

    return next_node


