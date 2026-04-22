"""
TSP Problem class: Integrates generator, evaluator, and prompts
"""

from typing import List, Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from problems.base import ProblemBase
from core.individual import Individual
from .generator import TSPInstanceGenerator
from .evaluator import TSPEvaluator
from .prompts import TSPPromptGenerator


class TSPProblem(ProblemBase):
    """TSP problem implementation"""

    def __init__(self, seed: int = 42):
        """
        Initialize TSP problem

        Args:
            seed: Random seed for training set generation
        """
        # Generate training set
        generator = TSPInstanceGenerator(seed=seed)
        training_set = generator.get_training_set()

        # Initialize base class
        super().__init__(name='TSP', training_set=training_set)

        self.evaluator = TSPEvaluator()
        self.prompt_generator = TSPPromptGenerator()

    def evaluate(self,
                 individual: Individual,
                 temp_dir: str,
                 timeout: int) -> Tuple[bool, float, float]:
        """
        Evaluate individual on TSP training set

        Args:
            individual: Individual to evaluate
            temp_dir: Temporary directory
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, fitness, execution_time)
        """
        return self.evaluator.execute_on_dataset(
            individual, self.training_set, temp_dir, timeout
        )

    def get_system_prompt(self) -> str:
        """Get system prompt for TSP"""
        return self.prompt_generator.get_system_prompt()

    def get_prompt_i1(self) -> str:
        """Get I1 operator prompt"""
        return self.prompt_generator.get_prompt_i1()

    def get_prompt_e1(self, individuals: List[Individual]) -> str:
        """Get E1 operator prompt"""
        return self.prompt_generator.get_prompt_e1(individuals)

    def get_prompt_e2(self, individuals: List[Individual]) -> str:
        """Get E2 operator prompt"""
        return self.prompt_generator.get_prompt_e2(individuals)

    def get_prompt_m1(self, individual: Individual) -> str:
        """Get M1 operator prompt"""
        return self.prompt_generator.get_prompt_m1(individual)

    def get_prompt_m2(self, individual: Individual) -> str:
        """Get M2 operator prompt"""
        return self.prompt_generator.get_prompt_m2(individual)

    def print_training_set_info(self) -> None:
        """Print TSP training set information"""
        for inst in self.training_set:
            print(f"  {inst['name']}: {len(inst['coords'])} cities")
