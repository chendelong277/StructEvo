"""
Base class for problem definitions
Defines the interface that all problems must implement
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import sys
import os

# Add parent directory to path to import core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.individual import Individual


class ProblemBase(ABC):
    """
    Abstract base class for optimization problems
    All specific problems (TSP, CVRP, etc.) must inherit from this
    """

    def __init__(self, name: str, training_set: List[Dict]):
        """
        Initialize problem

        Args:
            name: Problem name (e.g., 'TSP', 'CVRP')
            training_set: List of training instances
        """
        self.name = name
        self.training_set = training_set

    @abstractmethod
    def evaluate(self,
                 individual: Individual,
                 temp_dir: str,
                 timeout: int) -> Tuple[bool, float, float]:
        """
        Evaluate an individual on the training set

        Args:
            individual: Individual to evaluate
            temp_dir: Temporary directory for code execution
            timeout: Execution timeout in seconds

        Returns:
            Tuple of (success, fitness, execution_time)
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get system prompt for LLM

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_prompt_i1(self) -> str:
        """
        Get I1 operator prompt (Initialization)

        Returns:
            Prompt string
        """
        pass

    @abstractmethod
    def get_prompt_e1(self, individuals: List[Individual]) -> str:
        """
        Get E1 operator prompt (Explore different algorithm)

        Args:
            individuals: Selected individuals for reference

        Returns:
            Prompt string
        """
        pass

    @abstractmethod
    def get_prompt_e2(self, individuals: List[Individual]) -> str:
        """
        Get E2 operator prompt (Exploit common insights)

        Args:
            individuals: Selected individuals for reference

        Returns:
            Prompt string
        """
        pass

    @abstractmethod
    def get_prompt_m1(self, individual: Individual) -> str:
        """
        Get M1 operator prompt (Modify existing algorithm)

        Args:
            individual: Individual to modify

        Returns:
            Prompt string
        """
        pass

    @abstractmethod
    def get_prompt_m2(self, individual: Individual) -> str:
        """
        Get M2 operator prompt (Fine-tune parameters)

        Args:
            individual: Individual to fine-tune

        Returns:
            Prompt string
        """
        pass

    def print_training_set_info(self) -> None:
        """Print information about training set (can be overridden)"""
        for inst in self.training_set:
            print(f"  {inst['name']}: {inst}")
