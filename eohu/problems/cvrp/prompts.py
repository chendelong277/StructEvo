"""
CVRP prompt generator: Generates prompts for all EoH-U operators
"""

from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.individual import Individual


class CVRPPromptGenerator:
    """Generate prompts for CVRP problem"""

    TASK_DESCRIPTION = '''The Capacitated Vehicle Routing Problem (CVRP): Given a set of customers with locations and demands,
a central depot, and vehicles with limited capacity, find the minimum-cost routes to serve all customers while respecting capacity constraints.'''

    SYSTEM_PROMPT = '''You are an expert in operations research and optimization algorithms, specializing in solving the Capacitated Vehicle Routing Problem (CVRP).

Your task is to design a COMPLETE CVRP solving algorithm that finds the minimum-cost routes for a fleet of vehicles to serve all customers from a central depot while respecting vehicle capacity constraints.

Please provide your output in the following strict format:

<Algorithm Description>
A concise description of your algorithm's core idea (1-2 sentences).

<Code>
```python
import numpy as np
import random
from typing import List, Tuple

def solve_vrp(distance_matrix: np.ndarray, demands: List[int], capacity: int, depot_index: int = 0) -> List[List[int]]:
    """
    Solve the Capacitated Vehicle Routing Problem.

    Args:
        distance_matrix: (n, n) numpy array, distances between nodes
        demands: List of demands for each node (demands[depot_index] = 0)
        capacity: Maximum capacity of each vehicle
        depot_index: Index of the depot (default: 0)

    Returns:
        routes: List of routes, where each route is a list of node indices
                Example: [[0, 2, 3, 0], [0, 4, 1, 0]]
                Requirements:
                1. Each route must start and end at depot_index
                2. All non-depot nodes (customers) must be visited exactly once
                3. Total demand of each route must not exceed capacity
    """
    # Your complete algorithm implementation here

    return routes
```

REQUIREMENTS:
1. The code must complete within 120 seconds
2. Handle all edge cases correctly
3. Return valid routes respecting capacity constraints
4. All customers must be visited exactly once
5. Each route must start and end at the depot
6. Code must be syntactically correct with proper indentation
7. You may use numpy, random, math, and collections libraries
8. Ensure routes are feasible (sum of demands in each route <= capacity)
9. Design your own algorithm - it should be different from standard textbook algorithms
'''

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get system prompt"""
        return cls.SYSTEM_PROMPT

    @classmethod
    def get_prompt_i1(cls) -> str:
        """I1 Operator: Initialize with novel algorithm"""
        return f'''{cls.TASK_DESCRIPTION}

Please design a COMPLETE, NOVEL CVRP solving algorithm from scratch.

Your algorithm should be:
- Computationally efficient (must complete in <120 seconds)
- Properly handle capacity constraints


Please strictly follow the output format:
<Algorithm Description>
Your algorithm's core idea (1-2 sentences)

<Code>
Your complete implementation
'''

    @classmethod
    def get_prompt_e1(cls, individuals: List[Individual]) -> str:
        """E1 Operator: Explore completely different algorithm"""
        indivs_prompt = ''
        for i, ind in enumerate(individuals):
            indivs_prompt += f'''Algorithm {i + 1}:
Description: {ind.thought}
Fitness: {ind.fitness:.2f}

Code snippet:
{ind.code[:500]}...

'''
        return f'''{cls.TASK_DESCRIPTION}

Below are {len(individuals)} existing CVRP algorithms:

{indivs_prompt}

Please design a COMPLETELY DIFFERENT algorithm that:
- Uses a fundamentally different approach/mechanism
- Avoids similarity with the above algorithms
- Properly handles capacity constraints

Please strictly follow the output format:
<Algorithm Description>
Your algorithm's core idea (1-2 sentences)

<Code>
Your complete implementation
'''

    @classmethod
    def get_prompt_e2(cls, individuals: List[Individual]) -> str:
        """E2 Operator: Exploit common insights"""
        indivs_prompt = ''
        for i, ind in enumerate(individuals):
            indivs_prompt += f'''Algorithm {i + 1}:
Description: {ind.thought}
Fitness: {ind.fitness:.2f}

Code snippet:
{ind.code[:500]}...

'''
        return f'''{cls.TASK_DESCRIPTION}

Below are {len(individuals)} CVRP algorithms:

{indivs_prompt}

Please:
1. Analyze the common successful patterns/insights from these algorithms
2. Design a NEW algorithm that builds upon these insights
3. Add your own innovations while maintaining the proven strategies
4. Ensure capacity constraints are properly handled

Please strictly follow the output format:
<Algorithm Description>
Your algorithm's core idea (1-2 sentences)

<Code>
Your complete implementation
'''

    @classmethod
    def get_prompt_m1(cls, individual: Individual) -> str:
        """M1 Operator: Modify existing algorithm"""
        return f'''{cls.TASK_DESCRIPTION}

Below is an existing CVRP algorithm:

Description: {individual.thought}
Current Fitness: {individual.fitness:.2f}

Code:
{individual.code}

Please IMPROVE this algorithm by:
- Enhancing its core mechanisms
- Adding new optimization strategies
- Fixing potential weaknesses
- Improving capacity constraint handling
- Maintaining its fundamental approach while making it better

Please strictly follow the output format:
<Algorithm Description>
Your improved algorithm's core idea (1-2 sentences)

<Code>
Your complete implementation
'''

    @classmethod
    def get_prompt_m2(cls, individual: Individual) -> str:
        """M2 Operator: Fine-tune parameters"""
        return f'''{cls.TASK_DESCRIPTION}

Below is an existing CVRP algorithm:

Description: {individual.thought}
Current Fitness: {individual.fitness:.2f}

Code:
{individual.code}

Please FINE-TUNE this algorithm by:
- Adjusting numerical parameters
- Tuning hyperparameters
- Optimizing constants and thresholds
- Balancing exploration vs exploitation
- Fine-tuning capacity utilization strategies

Keep the core algorithm structure but optimize its parameters for better performance.

Please strictly follow the output format:
<Algorithm Description>
Your tuned algorithm's description (1-2 sentences)

<Code>
Your complete implementation
'''
