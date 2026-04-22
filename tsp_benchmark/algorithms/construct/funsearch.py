import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray,
                     distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]

    # Calculate distances from current node and to destination
    current_distances = distance_matrix[current_node, unvisited_nodes]
    dest_distances = distance_matrix[unvisited_nodes, destination_node]

    # Normalize both distance metrics
    norm_current = current_distances / np.max(current_distances)
    norm_dest = dest_distances / np.max(dest_distances)

    # Combine with weights (can be adjusted)
    combined_score = 0.6 * norm_current + 0.4 * (1 - norm_dest)  # Prefer smaller dest distances

    # Select node with minimum combined score
    next_node = unvisited_nodes[np.argmin(combined_score)]

    return next_node


