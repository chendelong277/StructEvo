import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]

    # Current proximity (harmonic to handle zero distances)
    current_dists = distance_matrix[current_node, unvisited_nodes]
    proximity = 1 / (current_dists + 1e-8)

    # Exact future potential via MST approximation
    future_potential = np.zeros(len(unvisited_nodes))
    for i, node in enumerate(unvisited_nodes):
        remaining_nodes = np.delete(unvisited_nodes, i)
        if not remaining_nodes.size:
            future_potential[i] = 0
            continue
        # Approximate remaining tour length using nearest neighbor distances
        remaining_dists = distance_matrix[node, remaining_nodes]
        future_potential[i] = np.mean(np.sort(remaining_dists)[:3])  # Top-3 nearest

    # Normalization with stability guarantees
    def safe_normalize(x):
        x = x - x.min()
        return x / (x.max() + 1e-8)

    p_norm = safe_normalize(proximity)
    fp_norm = safe_normalize(future_potential)

    # Adaptive weights using sigmoid transition
    progress = 1 - len(unvisited_nodes) / distance_matrix.shape[0]
    exploit_weight = 0.8 / (1 + np.exp(5 * (progress - 0.6)))  # Sigmoid centered at 60%

    explore_weight = 0.2 * (1 - progress)**2  # Quadratic decay

    # Combined score with directional exploration
    base_score = 0.6 * p_norm + 0.4 * fp_norm
    noise = np.random.normal(0, 0.05, len(unvisited_nodes)) * explore_weight
    combined_score = exploit_weight * base_score + noise
    return unvisited_nodes[np.argmax(combined_score)]