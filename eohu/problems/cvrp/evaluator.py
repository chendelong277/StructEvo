"""
CVRP evaluator: Validates and evaluates CVRP solutions
"""

import os
import time
import threading
import numpy as np
from typing import List, Tuple
from importlib.machinery import SourceFileLoader
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.individual import Individual


class CVRPEvaluator:
    """Evaluates CVRP solutions"""

    @staticmethod
    def check_vrp_solution(routes: List[List[int]],
                          demands: List[int],
                          capacity: int,
                          depot_index: int = 0) -> Tuple[bool, str]:
        """
        Verify CVRP solution validity

        Args:
            routes: List of routes
            demands: Node demands
            capacity: Vehicle capacity
            depot_index: Depot index

        Returns:
            Tuple of (is_valid, message)
        """
        if not routes:
            return False, "Empty solution"

        n_nodes = len(demands)
        visited_count = [0] * n_nodes

        # Check each route
        for r_idx, route in enumerate(routes):
            if not route:
                continue

            # Check start and end points
            if route[0] != depot_index or route[-1] != depot_index:
                return False, f"Route {r_idx} does not start/end at depot"

            # Check capacity constraint
            route_load = sum(demands[node] for node in route)
            if route_load > capacity:
                return False, f"Route {r_idx} exceeds capacity: load {route_load} > capacity {capacity}"

            # Record visits
            for node in route:
                if node != depot_index:
                    if node < 0 or node >= n_nodes:
                        return False, f"Node index {node} out of range"
                    visited_count[node] += 1

        # Check customer coverage
        for i in range(n_nodes):
            if i == depot_index:
                continue

            if visited_count[i] == 0:
                return False, f"Customer {i} not visited"
            elif visited_count[i] > 1:
                return False, f"Customer {i} visited multiple times (count: {visited_count[i]})"

        return True, "Valid CVRP solution"

    @staticmethod
    def calculate_vrp_total_length(routes: List[List[int]],
                                   distance_matrix: np.ndarray) -> float:
        """
        Calculate total distance of all routes

        Args:
            routes: List of routes
            distance_matrix: Distance matrix

        Returns:
            Total distance
        """
        total_dist = 0.0
        for route in routes:
            for i in range(len(route) - 1):
                u = route[i]
                v = route[i + 1]
                total_dist += distance_matrix[u][v]
        return float(total_dist)

    @staticmethod
    def execute_on_dataset(individual: Individual,
                          training_set: List[dict],
                          temp_dir: str,
                          timeout: int = 120) -> Tuple[bool, float, float]:
        """
        Execute individual on multiple CVRP training instances

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
            def run_solver(dist_mat, demands, capacity, depot):
                solver_module = SourceFileLoader(module_name, temp_file).load_module()
                if not hasattr(solver_module, 'solve_vrp'):
                    raise AttributeError('solve_vrp function not defined in code')
                return solver_module.solve_vrp(dist_mat, demands, capacity, depot)

            total_fitness = 0
            total_time = 0
            success_count = 0
            individual.fitness_per_instance = {}

            for instance in training_set:
                instance_name = instance['name']
                distance_matrix = instance['distance_matrix']
                demands = instance['demands']
                capacity = instance['capacity']
                depot_index = instance['depot_index']

                try:
                    start_time = time.time()
                    routes = run_solver(distance_matrix, demands, capacity, depot_index)
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    # Validate solution
                    is_valid, message = CVRPEvaluator.check_vrp_solution(routes, demands, capacity, depot_index)

                    if not is_valid:
                        print(f'  [{instance_name}] Failed: {message}')
                        individual.fitness_per_instance[instance_name] = float('inf')
                        continue

                    # Calculate fitness
                    true_fitness = CVRPEvaluator.calculate_vrp_total_length(routes, distance_matrix)
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
