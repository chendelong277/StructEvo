"""
TSP instance generator
Generates multi-distribution, multi-scale TSP training sets
"""

import random
import numpy as np
from typing import List, Tuple, Dict


class TSPInstanceGenerator:
    """Generate TSP training instances with different distributions"""

    def __init__(self, seed: int = 42):
        """
        Initialize generator

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_uniform(self, n: int) -> List[Tuple[float, float]]:
        """
        Generate uniform random distribution

        Args:
            n: Number of nodes

        Returns:
            List of (x, y) coordinates
        """
        return [(float(x), float(y)) for x, y in np.random.rand(n, 2)]

    def generate_clustered(self, n: int, num_clusters: int = 5) -> List[Tuple[float, float]]:
        """
        Generate clustered distribution

        Args:
            n: Number of nodes
            num_clusters: Number of clusters

        Returns:
            List of (x, y) coordinates
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
        return [(float(x), float(y)) for x, y in points]

    def generate_explosion(self, n: int) -> List[Tuple[float, float]]:
        """
        Generate explosion (Gaussian) distribution

        Args:
            n: Number of nodes

        Returns:
            List of (x, y) coordinates
        """
        points = np.random.normal(loc=0.5, scale=0.15, size=(n, 2))
        points = np.clip(points, 0, 1)
        return [(float(x), float(y)) for x, y in points]

    def get_training_set(self) -> List[Dict]:
        """
        Generate complete training set

        Returns:
            List of training instances
        """
        dataset = []
        dataset.append({'name': 'Uniform_50', 'coords': self.generate_uniform(50)})
        dataset.append({'name': 'Clustered_50', 'coords': self.generate_clustered(50)})
        dataset.append({'name': 'Explosion_100', 'coords': self.generate_explosion(100)})
        return dataset
