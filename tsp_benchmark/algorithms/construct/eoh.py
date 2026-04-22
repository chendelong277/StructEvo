import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]

    momentum = np.sum(distance_matrix[unvisited_nodes] - distance_matrix[current_node, unvisited_nodes].reshape(-1, 1), axis=1)
    degree_attraction = np.sum(distance_matrix[unvisited_nodes] > 0, axis=1)
    cluster_penalty = -np.mean(distance_matrix[:, unvisited_nodes], axis=0)
    exploitation = np.random.rand(len(unvisited_nodes))

    remaining_nodes = len(unvisited_nodes)
    total_nodes = len(distance_matrix)
    progress = remaining_nodes / total_nodes

    momentum_weight = np.tanh(progress)
    attraction_weight = 1 / (1 + np.exp(-5 * (1 - progress)))
    penalty_weight = np.tanh(1 - progress)
    exploitation_weight = 0.1 * progress

    combined_score = (momentum_weight * momentum) + (attraction_weight * degree_attraction) + (penalty_weight * cluster_penalty) + (exploitation_weight * exploitation)
    return unvisited_nodes[np.argmax(combined_score)]