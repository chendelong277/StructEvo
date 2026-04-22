import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited_nodes = np.array(unvisited_nodes)
    unvisited = unvisited_nodes[unvisited_nodes != current_node]

    if len(unvisited) == 0:
        return destination_node

    def compute_scores():
        scores = []
        sub_distance_matrix = distance_matrix[np.ix_(unvisited, unvisited)]

        max_dist = np.max(sub_distance_matrix)
        density_eps = max_dist / len(unvisited) if len(unvisited) > 0 else 1.0
        if density_eps == 0: density_eps = 1e-5

        cluster_labels = DBSCAN(eps=density_eps, min_samples=1, metric='precomputed').fit_predict(sub_distance_matrix)

        total_unvisited = len(unvisited)

        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        entropy_val = entropy(counts / total_unvisited)

        mean_distances = np.mean(sub_distance_matrix, axis=1)

        for i, node in enumerate(unvisited):
            distance_to_node = distance_matrix[current_node][node]

            if distance_to_node <= 1e-9:
                inv_distance = 1e6
            else:
                inv_distance = 1.0 / distance_to_node

            connectivity_score = np.sum(sub_distance_matrix[i] < np.inf) - 1
            variance_score = np.var(sub_distance_matrix[i])
            dynamic_decay = (1 / (np.log(1 + mean_distances[i] + 1) + 1))

            score = inv_distance + (0.1 * connectivity_score * dynamic_decay) - (0.05 * variance_score) - (0.1 * entropy_val)
            scores.append(score)

        return scores

    scores = compute_scores()

    if not scores:
        return destination_node

    next_node = unvisited[np.argmax(scores)]

    return next_node