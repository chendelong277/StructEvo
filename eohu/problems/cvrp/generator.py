"""
CVRP instance generator
Generates multi-distribution, multi-scale CVRP training sets
"""

import random
import math
import numpy as np
from typing import List, Tuple, Dict


class CVRPInstanceGenerator:
    """Generate CVRP training instances with different distributions"""

    def __init__(self, seed: int = 42):
        """
        Initialize generator

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_uniform(self, n: int) -> Tuple[List[Tuple[float, float]], List[int], int]:
        """
        Generate uniform random distribution with demands

        Args:
            n: Number of nodes

        Returns:
            Tuple of (coordinates, demands, capacity)
        """
        coords = np.random.rand(n, 2)
        coordinates = [(float(x), float(y)) for x, y in coords]

        demands = [0] + [random.randint(5, 20) for _ in range(n - 1)]
        total_demand = sum(demands)
        capacity = max(50, int(total_demand * 0.3))
        return coordinates, demands, capacity

    def generate_clustered(self, n: int, num_clusters: int = 5) -> Tuple[List[Tuple[float, float]], List[int], int]:
        """
        Generate clustered distribution with demands

        Args:
            n: Number of nodes
            num_clusters: Number of clusters

        Returns:
            Tuple of (coordinates, demands, capacity)
        """
        centers = np.random.rand(num_clusters, 2)
        points = []
        points_per_cluster = n // num_clusters

        for i in range(num_clusters):
            cluster_points = np.random.normal(loc=centers[i], scale=0.04, size=(points_per_cluster, 2))
            points.extend(cluster_points)

        remaining = n - len(points)
        if remaining > 0:
            points.extend(np.random.rand(remaining, 2))

        points = np.array(points)
        points = np.clip(points, 0, 1)
        coordinates = [(float(x), float(y)) for x, y in points]

        demands = [0] + [random.randint(5, 20) for _ in range(n - 1)]
        total_demand = sum(demands)
        capacity = max(50, int(total_demand * 0.3))
        return coordinates, demands, capacity

    def generate_explosion(self, n: int) -> Tuple[List[Tuple[float, float]], List[int], int]:
        """
        Generate explosion (Gaussian) distribution with demands

        Args:
            n: Number of nodes

        Returns:
            Tuple of (coordinates, demands, capacity)
        """
        points = np.random.normal(loc=0.5, scale=0.15, size=(n, 2))
        points = np.clip(points, 0, 1)
        coordinates = [(float(x), float(y)) for x, y in points]

        demands = [0] + [random.randint(5, 20) for _ in range(n - 1)]
        total_demand = sum(demands)
        capacity = max(100, int(total_demand * 0.3))
        return coordinates, demands, capacity

    def compute_distance_matrix(self, coordinates: List[Tuple[float, float]]) -> np.ndarray:
        """
        Compute distance matrix using Euclidean distance

        Args:
            coordinates: List of coordinates

        Returns:
            Distance matrix
        """
        n = len(coordinates)
        distance_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = coordinates[i]
                    x2, y2 = coordinates[j]
                    distance_matrix[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance_matrix

    def get_training_set(self) -> List[Dict]:
        """
        Generate complete training set

        Returns:
            List of training instances
        """
        dataset = []

        # Instance 1: Uniform_50
        coords_u, demands_u, capacity_u = self.generate_uniform(50)
        dist_u = self.compute_distance_matrix(coords_u)
        dataset.append({
            'name': 'Uniform_50',
            'coordinates': coords_u,
            'demands': demands_u,
            'capacity': capacity_u,
            'distance_matrix': dist_u,
            'depot_index': 0
        })

        # Instance 2: Clustered_50
        coords_c, demands_c, capacity_c = self.generate_clustered(50)
        dist_c = self.compute_distance_matrix(coords_c)
        dataset.append({
            'name': 'Clustered_50',
            'coordinates': coords_c,
            'demands': demands_c,
            'capacity': capacity_c,
            'distance_matrix': dist_c,
            'depot_index': 0
        })

        # Instance 3: Explosion_100
        coords_e, demands_e, capacity_e = self.generate_explosion(100)
        dist_e = self.compute_distance_matrix(coords_e)
        dataset.append({
            'name': 'Explosion_100',
            'coordinates': coords_e,
            'demands': demands_e,
            'capacity': capacity_e,
            'distance_matrix': dist_e,
            'depot_index': 0
        })

        return dataset
