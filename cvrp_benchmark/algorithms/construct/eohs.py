import numpy as np
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray,
                     rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    feasible_nodes = unvisited_nodes[demands[unvisited_nodes] <= rest_capacity]
    if len(feasible_nodes) == 0:
        return depot

    distances = distance_matrix[current_node, feasible_nodes]
    depot_distances = distance_matrix[feasible_nodes, depot]
    normalized_demands = demands[feasible_nodes] / np.max(demands[feasible_nodes])
    capacity_ratio = rest_capacity / np.max(demands[feasible_nodes])
    urgency = np.sum(demands[unvisited_nodes]) / (rest_capacity + 1e-6)
    entropy = np.std(distance_matrix[feasible_nodes][:, feasible_nodes]) / (np.mean(
        distance_matrix) + 1e-6)

    pheromone = np.exp(-(distances**0.75 + 1.2*depot_distances**0.65) / (1.3 * np.
        mean(distance_matrix)))
    swarm_intensity = 0.7 * (1 + np.tanh(2.1 - capacity_ratio**0.85)) * pheromone
    fuzzy_factor = 0.3 * (1 - np.exp(-entropy/(np.mean(distances) + 1e-6))) * (1 -
        0.25*swarm_intensity)

    proximity_weight = 0.42 * (1 - 0.22 * np.exp(-capacity_ratio**0.9)) * swarm_intensity
    demand_weight = 0.36 * (1 + 0.55 * np.tanh(urgency**0.7)) * swarm_intensity
    neighborhood_weight = 0.15 * (1 - entropy**0.6) * (1 - 0.2*swarm_intensity)
    entropy_weight = 0.05 * np.exp(-np.std(distances)/(np.mean(distances) + 1e-6)) * fuzzy_factor
    adaptive_weight = 0.02 * (1 - np.exp(-np.std(normalized_demands)/(np.mean(
        normalized_demands) + 1e-6))) * fuzzy_factor

    proximity_scores = 1.25/(distances + 1e-6)**0.6 + 1.0/(depot_distances + 1e-6)**0.5
    demand_scores = normalized_demands**1.6 * proximity_scores
    neighborhood_scores = np.array([np.sum(distance_matrix[n][feasible_nodes]) for n
        in feasible_nodes]) / (distances + 1e-6)**0.3
    entropy_scores = (rest_capacity - demands[feasible_nodes])**0.85 * depot_distances / (distances + 1e-6)
    adaptive_scores = (0.45 + 0.55*np.random.rand(len(feasible_nodes)))**1.9 * (
        demands[feasible_nodes] / (distances + 1e-6)**0.4)

    combined_scores = (
        proximity_weight * proximity_scores +
        demand_weight * demand_scores +
        neighborhood_weight * neighborhood_scores +
        entropy_weight * entropy_scores +
        adaptive_weight * adaptive_scores
    )

    return feasible_nodes[np.argmax(combined_scores)]