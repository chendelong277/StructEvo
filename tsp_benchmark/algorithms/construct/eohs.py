import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    current_dist = distance_matrix[current_node, unvisited_nodes]
    dest_dist = distance_matrix[destination_node, unvisited_nodes]
    progress = distance_matrix[current_node, destination_node] - dest_dist
    exploration_factor = np.log(len(unvisited_nodes) + 1) * (1 + np.random.rand() * 0.5)
    centrality = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)
    penalty = np.maximum(0, dest_dist - np.percentile(dest_dist, 85))
    k = min(5, len(unvisited_nodes) - 1)
    if k > 0:
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        cluster_novelty = -np.mean(np.partition(sub_matrix, k, axis=1)[:, :k], axis=1)
    else:
        cluster_novelty = np.zeros(len(unvisited_nodes))
    if len(unvisited_nodes) < len(distance_matrix) - 1:
        last_move_dir = distance_matrix[unvisited_nodes, current_node] - distance_matrix[unvisited_nodes, destination_node]
        momentum = np.abs(last_move_dir - np.mean(last_move_dir)) * (1 + 0.1 * np.random.rand())
    else:
        momentum = np.zeros(len(unvisited_nodes))
    path_diversity = np.std(distance_matrix[unvisited_nodes], axis=1) * (1 + 0.1 * np.random.rand())
    remaining_path_heuristic = np.mean(distance_matrix[unvisited_nodes], axis=1) * (1 - 0.1 * np.random.rand())
    phase = len(unvisited_nodes) / len(distance_matrix)
    scale_factor = np.mean(distance_matrix) / np.max(distance_matrix)
    entropy = -np.sum(np.exp(-current_dist) * np.log(np.exp(-current_dist) + 1e-10))
    entropy_factor = 0.1 * entropy * (1 - phase)
    w_dist = (0.25 + (0.05 * phase)) * scale_factor
    w_progress = (0.15 - (0.05 * phase)) * scale_factor
    w_explore = (0.15 - (0.05 * phase)) * (1 - scale_factor) + entropy_factor
    w_centrality = (0.1 + (0.05 * phase)) * scale_factor
    w_penalty = (0.1 - (0.05 * phase)) * scale_factor
    w_novelty = (0.1 + (0.05 * phase)) * (1 - scale_factor)
    w_momentum = (0.05 * (1 - phase)) * scale_factor
    w_diversity = (0.05 * (1 - phase)) * (1 - scale_factor)
    w_heuristic = (0.05 * phase) * (1 - scale_factor)
    score = (w_dist * current_dist) + (w_progress * progress) + (w_explore * exploration_factor) - (w_centrality * centrality) + (w_penalty * penalty) + (w_novelty * cluster_novelty) + (w_momentum * momentum) + (w_diversity * path_diversity) + (w_heuristic * remaining_path_heuristic)
    return unvisited_nodes[np.argmin(score)]