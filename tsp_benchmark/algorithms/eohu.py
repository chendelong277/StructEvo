
import numpy as np
from typing import List, Tuple
import math

def solve_tsp(coordinates: List[Tuple[float, float]]) -> Tuple[float, List[int]]:

    if not coordinates:
        return 0.0, []

    n = len(coordinates)
    if n == 1:
        return 0.0, [0]

    points = np.array(coordinates)

    center = np.mean(points, axis=0)
    vectors = points - center
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    distances = np.linalg.norm(vectors, axis=1)

    spiral_order = sorted(range(n), key=lambda i: (angles[i], distances[i]))

    improved = True
    max_iterations = min(1000, n * 5)
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(n):
            for j in range(i + 2, min(i + n - 1, n)):
                a, b = spiral_order[i], spiral_order[(i + 1) % n]
                c, d = spiral_order[j], spiral_order[(j + 1) % n]

                current = (np.linalg.norm(points[a] - points[b]) +
                           np.linalg.norm(points[c] - points[d]))
                proposed = (np.linalg.norm(points[a] - points[c]) +
                            np.linalg.norm(points[b] - points[d]))

                if proposed < current:
                    spiral_order[i + 1:j + 1] = spiral_order[i + 1:j + 1][::-1]
                    improved = True

    route = spiral_order + [spiral_order[0]]
    total_distance = sum(np.linalg.norm(points[route[i]] - points[route[i + 1]])
                         for i in range(n))

    return total_distance, spiral_order