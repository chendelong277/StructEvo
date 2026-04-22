import numpy as np


def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray,
                     rest_capacity: np.ndarray, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    if not unvisited_nodes.size:
        return depot

    # Adaptive capacity buffer with demand characteristics and route progress
    demand_stats = demands[unvisited_nodes]
    demand_cv = np.std(demand_stats) / (np.mean(demand_stats) + 1e-10)
    route_progress = 1 - len(unvisited_nodes) / len(demands)

    # Three-component adaptive buffer
    buffer = (0.02 + 0.06 * (1 - np.exp(-3 * (demand_cv - 0.3))) + 0.03 *
              route_progress * (1 + 0.5 * demand_cv))

    feasible_mask = demands[unvisited_nodes] <= rest_capacity * (1 + buffer)
    feasible_nodes = unvisited_nodes[feasible_mask]

    if not feasible_nodes.size:
        return depot

    # Robust distance metrics using percentile normalization
    current_dists = distance_matrix[current_node, feasible_nodes]
    depot_dists = distance_matrix[feasible_nodes, depot]
    dist_p90 = np.percentile(distance_matrix, 90)

    # Normalized components with outlier protection
    norm_current = (current_dists - np.min(current_dists)) / (np.ptp(current_dists) + 1e-10)
    norm_depot = (depot_dists - np.min(depot_dists)) / (np.ptp(depot_dists) + 1e-10)

    # Route state analysis with multiple factors
    remaining_demand = np.sum(demands[unvisited_nodes])
    capacity_ratio = min(1.0, remaining_demand / (rest_capacity + 1e-10))
    urgency = (len(unvisited_nodes) / len(demands)) ** 0.7

    # Core scoring components with enhanced formulations
    proximity = (0.8 / (current_dists + 0.1 * dist_p90) +
                 0.2 * (1 - norm_current) * (1 + 0.3 * urgency))
    utilization = np.power(demands[feasible_nodes], 0.8) / (rest_capacity + 1e-10)
    spatial_cohesion = np.exp(-3 * abs(norm_current - (1 - norm_depot)))

    # Dynamic weight adaptation using route state
    proximity_weight = 0.7 - 0.2 * (1 / (1 + np.exp(-12 * (capacity_ratio - 0.4))))
    utilization_weight = 0.6 / (1 + np.exp(-10 * (1.2 - capacity_ratio)))
    spatial_weight = 0.4 * (1 - np.exp(-3 * urgency))

    # Advanced spatial clustering analysis
    if len(feasible_nodes) > 1:
        centroid = np.mean(distance_matrix[feasible_nodes], axis=0)
        spatial_scores = np.linalg.norm(distance_matrix[feasible_nodes] - centroid, axis=1)
        spatial_scores = (spatial_scores - np.min(spatial_scores)) / (np.ptp(spatial_scores) + 1e-10)
    else:
        spatial_scores = np.zeros(len(feasible_nodes))

    # Adaptive critical demand detection
    demand_ratio = demands[feasible_nodes] / rest_capacity
    critical_threshold = (0.5 + 0.3 * np.tanh(5 * (demand_cv - 0.4)) + 0.15 *
                          route_progress)
    critical_bonus = np.where(demand_ratio > critical_threshold,
                              np.maximum(0, demand_ratio - critical_threshold) ** 1.5, 0)

    # Balanced composite scoring
    scores = (proximity_weight * proximity + utilization_weight * utilization +
              spatial_weight * spatial_cohesion + 0.7 * spatial_scores + 1.5 *
              critical_bonus)

    # Sophisticated tie-breaking mechanism
    best_idx = np.argmax(scores)
    if np.sum(np.isclose(scores, scores[best_idx], rtol=1e-8, atol=1e-8)) > 1:
        tied_nodes = feasible_nodes[np.isclose(scores, scores[best_idx])]
        tie_breakers = np.column_stack([
            -critical_bonus[np.isclose(scores, scores[best_idx])],
            -spatial_cohesion[np.isclose(scores, scores[best_idx])],
            current_dists[np.isclose(scores, scores[best_idx])],
            -utilization[np.isclose(scores, scores[best_idx])]
        ])
        return tied_nodes[np.lexsort(tie_breakers.T)[0]]

    return feasible_nodes[best_idx]