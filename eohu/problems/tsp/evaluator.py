"""
TSP evaluator: Validates and evaluates TSP solutions
"""

import os
import time
import threading
from typing import List, Tuple
from importlib.machinery import SourceFileLoader
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.individual import Individual


class TSPEvaluator:
    """Evaluates TSP solutions"""

    @staticmethod
    def check_path(coordinates: List[Tuple[float, float]], route: List[int]) -> bool:
        """
        Check if route is valid

        Args:
            coordinates: City coordinates
            route: Route (list of node indices)

        Returns:
            True if valid, False otherwise
        """
        n = len(coordinates)
        if not route:
            return False
        if len(route) != n:
            return False

        visited = [False] * n
        for node in route:
            if not isinstance(node, int):
                return False
            if node < 0 or node >= n:
                return False
            if visited[node]:
                return False
            visited[node] = True

        return all(visited)

    @staticmethod
    def calculate_path_length(coordinates: List[Tuple[float, float]], route: List[int]) -> float:
        """
        Calculate total path length

        Args:
            coordinates: City coordinates
            route: Route (list of node indices)

        Returns:
            Total path length
        """
        total_length = 0
        n = len(route)
        for i in range(n):
            start_node = route[i]
            end_node = route[(i + 1) % n]
            start_coord = coordinates[start_node]
            end_coord = coordinates[end_node]
            total_length += ((start_coord[0] - end_coord[0]) ** 2 +
                            (start_coord[1] - end_coord[1]) ** 2) ** 0.5
        return total_length

    @staticmethod
    def execute_on_dataset(individual: Individual,
                          training_set: List[dict],
                          temp_dir: str,
                          timeout: int = 60) -> Tuple[bool, float, float]:
        """
        Execute individual on multiple training instances

        Args:
            individual: Individual to evaluate
            training_set: List of training instances
            temp_dir: Temporary directory for code execution
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, average_fitness, average_time)
        """
        class TimeoutException(Exception):
            pass

        def time_limit(seconds):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    result = [TimeoutException('Function call timed out')]
                    def target():
                        try:
                            result[0] = func(*args, **kwargs)
                        except Exception as e:
                            result[0] = e
                    thread = threading.Thread(target=target)
                    thread.start()
                    thread.join(seconds)
                    if thread.is_alive():
                        raise TimeoutException('Function call timed out')
                    if isinstance(result[0], BaseException):
                        raise result[0]
                    return result[0]
                return wrapper
            return decorator

        thread_id = threading.get_ident()
        temp_file_name = f'temp_exec_{thread_id}_{time.time()}.py'
        module_name = f'exec_{thread_id}_{int(time.time()*1000)}'
        temp_file = os.path.join(temp_dir, temp_file_name)

        try:
            with open(temp_file, 'w', encoding='utf-8') as file:
                file.write(individual.code)

            @time_limit(timeout)
            def run_solver(coords):
                solver_module = SourceFileLoader(module_name, temp_file).load_module()
                if not hasattr(solver_module, 'solve_tsp'):
                    raise AttributeError('solve_tsp function not defined in code')
                return solver_module.solve_tsp(coords)

            total_fitness = 0
            total_time = 0
            success_count = 0
            individual.fitness_per_instance = {}

            for instance in training_set:
                instance_name = instance['name']
                coordinates = instance['coords']

                try:
                    start_time = time.time()
                    route_length_reported, route_order = run_solver(coordinates)
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    if not TSPEvaluator.check_path(coordinates, route_order):
                        print(f'  [{instance_name}] Failed: Invalid path')
                        individual.fitness_per_instance[instance_name] = float('inf')
                        continue

                    true_fitness = TSPEvaluator.calculate_path_length(coordinates, route_order)
                    individual.fitness_per_instance[instance_name] = true_fitness
                    total_fitness += true_fitness
                    total_time += elapsed_time
                    success_count += 1

                except TimeoutException:
                    print(f'  [{instance_name}] Timeout')
                    individual.fitness_per_instance[instance_name] = float('inf')
                except Exception as e:
                    print(f'  [{instance_name}] Error: {type(e).__name__}')
                    individual.fitness_per_instance[instance_name] = float('inf')

            if success_count < len(training_set):
                print(f'  Passed {success_count}/{len(training_set)} instances - Failed')
                return False, float('inf'), 0.0

            avg_fitness = total_fitness / success_count
            avg_time = total_time / success_count

            print(f'  Average Fitness={avg_fitness:.2f} ({success_count}/{len(training_set)} success)')
            return True, avg_fitness, avg_time

        except Exception as e:
            print(f'  Execution failed: {type(e).__name__}')
            return False, float('inf'), 0.0
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
