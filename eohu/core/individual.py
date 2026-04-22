"""
Individual class: Represents an algorithm individual in the population
"""

from typing import Dict


class Individual:
    """Represents an algorithm individual with thought, code, and fitness"""

    def __init__(self, thought: str, code: str):
        """
        Initialize an individual

        Args:
            thought: Algorithm description/thought process
            code: Complete algorithm implementation code
        """
        self.thought = thought
        self.code = code
        self.fitness = float('inf')
        self.fitness_per_instance: Dict[str, float] = {}
        self.generation = 0
        self.execution_time = 0.0
        self.sample_id = 0

    def __repr__(self):
        return f'Individual(fitness={self.fitness:.2f}, gen={self.generation})'

    def is_valid(self) -> bool:
        """Check if individual has valid fitness"""
        return self.fitness != float('inf')

    def to_dict(self) -> dict:
        """Convert individual to dictionary for serialization"""
        return {
            'thought': self.thought,
            'code': self.code,
            'fitness': self.fitness,
            'fitness_per_instance': self.fitness_per_instance,
            'generation': self.generation,
            'execution_time': self.execution_time,
            'sample_id': self.sample_id
        }
