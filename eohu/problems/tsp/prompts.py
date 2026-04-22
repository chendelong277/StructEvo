"""
TSP prompt generator: Generates prompts for all EoH-U operators
"""

from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.individual import Individual


class TSPPromptGenerator:
    """Generate prompts for TSP problem"""

    TASK_DESCRIPTION = '''The Traveling Salesman Problem (TSP): Given a set of cities with coordinates,
find the shortest tour that visits each city exactly once and returns to the starting point.'''

    SYSTEM_PROMPT = '''You are an expert in operations research and optimization algorithms, specializing in solving the Traveling Salesman Problem (TSP).

Your task is to design a COMPLETE TSP solving algorithm that finds the shortest tour visiting all cities exactly once and returning to the starting point.

Please provide your output in the following strict format:

<Algorithm Description>
A concise description of your algorithm's core idea (1-2 sentences).

<Code>
```python
import numpy as np
from typing import List, Tuple

def solve_tsp(coordinates: List[Tuple[float, float]]) -> Tuple[float, List[int]]:
    """
    Solve the Traveling Salesman Problem.

    Args:
        coordinates: List of (x, y) tuples representing city locations

    Returns:
        Tuple of (route_length, route_order)
        - route_length: Total length of the tour
        - route_order: List of city indices representing the tour order
    """
    # Your complete algorithm implementation here

    return route_length, route_order
```

REQUIREMENTS:
1. The code must complete within 60 seconds
2. Total evaluations should not exceed 10,000
3. Avoid excessive recursion depth
4. Handle all edge cases correctly
5. Return a valid tour visiting all cities exactly once
6. Code must be syntactically correct with proper indentation
7. You may use numpy library
8. Design your own algorithm - it should be different from standard textbook algorithms
'''

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get system prompt"""
        return cls.SYSTEM_PROMPT

    @classmethod
    def get_prompt_i1(cls) -> str:
        """I1 Operator: Initialize with novel algorithm"""
        return f'''{cls.TASK_DESCRIPTION}

Please design a COMPLETE, NOVEL TSP solving algorithm from scratch.

Your algorithm should be:
- Fundamentally different from common approaches (nearest neighbor, genetic algorithm, simulated annealing, etc.)
- Creative and innovative in its approach
- Computationally efficient (must complete in <60 seconds)
- Able to handle instances with 50-100 cities

Consider exploring:
- Novel geometric insights
- Unique construction heuristics
- Creative improvement strategies
- Hybrid approaches with original combinations

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

Below are {len(individuals)} existing TSP algorithms:

{indivs_prompt}

Please design a COMPLETELY DIFFERENT algorithm that:
- Uses a fundamentally different approach/mechanism
- Avoids similarity with the above algorithms
- Explores a new region of the algorithm design space

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

Below are {len(individuals)} TSP algorithms:

{indivs_prompt}

Please:
1. Analyze the common successful patterns/insights from these algorithms
2. Design a NEW algorithm that builds upon these insights
3. Add your own innovations while maintaining the proven strategies

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

Below is an existing TSP algorithm:

Description: {individual.thought}
Current Fitness: {individual.fitness:.2f}

Code:
{individual.code}

Please IMPROVE this algorithm by:
- Enhancing its core mechanisms
- Adding new optimization strategies
- Fixing potential weaknesses
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

Below is an existing TSP algorithm:

Description: {individual.thought}
Current Fitness: {individual.fitness:.2f}

Code:
{individual.code}

Please FINE-TUNE this algorithm by:
- Adjusting numerical parameters
- Tuning hyperparameters
- Optimizing constants and thresholds
- Balancing exploration vs exploitation

Keep the core algorithm structure but optimize its parameters for better performance.

Please strictly follow the output format:
<Algorithm Description>
Your tuned algorithm's description (1-2 sentences)

<Code>
Your complete implementation
'''
