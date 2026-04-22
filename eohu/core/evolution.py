"""
EoH-U Evolution Framework: Main evolution controller
"""

import os
import time
import json
import re
import threading
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from .individual import Individual
from .population import Population
from .utils import extract_thought_and_code
from problems.base import ProblemBase


class EoHU_Evolution:
    """
    Evolution of Heuristics - Unrestricted (EoH-U)
    Problem-agnostic evolution framework
    """

    def __init__(self,
                 llm_client: OpenAI,
                 llm_model: str,
                 problem: ProblemBase,
                 pop_size: int = 5,
                 max_generations: int = 10,
                 max_sample_nums: int = 100,
                 selection_num: int = 2,
                 num_samplers: int = 4,
                 base_path: str = 'run_eohu',
                 timeout: int = 120):
        """
        Initialize EoH-U evolution framework

        Args:
            llm_client: OpenAI client instance
            llm_model: Model name to use
            problem: Problem instance (TSP, CVRP, etc.)
            pop_size: Population size
            max_generations: Maximum number of generations
            max_sample_nums: Maximum number of samples
            selection_num: Number of individuals for selection operators
            num_samplers: Number of parallel samplers
            base_path: Base directory for saving results
            timeout: Timeout for individual execution (seconds)
        """
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.problem = problem

        self.pop_size = pop_size
        self.max_generations = max_generations
        self.max_sample_nums = max_sample_nums
        self.selection_num = selection_num
        self.num_samplers = num_samplers
        self.base_path = base_path
        self.timeout = timeout

        # Create population
        self.population = Population(pop_size, base_path=base_path)

        # Statistics
        self.total_samples = 0
        self.lock = threading.Lock()

        # Initial sampling limit
        self.initial_sample_nums_max = min(self.max_sample_nums, 2 * self.pop_size)

        # Create output directories
        os.makedirs(base_path, exist_ok=True)
        self.temp_dir = os.path.join(base_path, 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)

        # Best tracking
        self.global_best_fitness = float('inf')
        self.global_best_individual = None

    def _sample_evaluate_register(self, prompt: str, operator: str) -> None:
        """
        Sample -> Evaluate -> Register pipeline

        Args:
            prompt: Prompt for LLM
            operator: Operator name (I1, E1, E2, M1, M2)
        """
        with self.lock:
            self.total_samples += 1
            sample_id = self.total_samples

        print(f"\n[{operator}] Sample #{sample_id}")

        individual = self._generate_individual(
            prompt=prompt,
            generation=self.population.generation,
            sample_id=sample_id,
            temperature=0.8,
            max_retries=2
        )

        # Register
        self.population.register_function(individual)

        # Update global best
        if individual.fitness < self.global_best_fitness:
            self.global_best_fitness = individual.fitness
            self.global_best_individual = individual
            print(f"  New global best: {self.global_best_fitness:.2f}")
            self._update_evolution_summary()

    def _generate_individual(self,
                            prompt: str,
                            generation: int,
                            sample_id: int,
                            temperature: float = 0.8,
                            max_retries: int = 2) -> Individual:
        """
        Generate and evaluate an individual algorithm

        Args:
            prompt: Prompt for LLM
            generation: Current generation number
            sample_id: Sample ID
            temperature: LLM temperature
            max_retries: Maximum retry attempts

        Returns:
            Individual instance
        """
        for attempt in range(max_retries):
            try:
                # Call LLM
                completion = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {'role': 'system', 'content': self.problem.get_system_prompt()},
                        {'role': 'user', 'content': prompt},
                    ],
                    temperature=temperature,
                )
                text = completion.choices[0].message.content

                # Extract thought and code
                thought, code = extract_thought_and_code(text)

                if not code or not thought:
                    print(f"  Extraction failed (attempt {attempt+1}/{max_retries})")
                    continue

                # Create individual
                individual = Individual(thought, code)
                individual.generation = generation
                individual.sample_id = sample_id

                print(f"  Algorithm: {thought[:60]}...")

                # Evaluate
                success, fitness, exec_time = self.problem.evaluate(
                    individual, self.temp_dir, self.timeout
                )

                individual.fitness = fitness
                individual.execution_time = exec_time

                if success:
                    return individual
                else:
                    if attempt < max_retries - 1:
                        print(f"  Evaluation failed, retrying ({attempt+1}/{max_retries})...")
                    continue

            except Exception as e:
                print(f'  Generation exception: {type(e).__name__}')
                if attempt < max_retries - 1:
                    continue

        print(f'  All {max_retries} attempts failed')
        ind = Individual('Failed', '# Error')
        ind.fitness = float('inf')
        ind.sample_id = sample_id
        return ind

    def _continue_loop(self) -> bool:
        """Check if evolution should continue"""
        return (self.population.generation < self.max_generations and
                self.total_samples < self.max_sample_nums)

    def _iteratively_init_population(self) -> None:
        """Initialization phase using I1 operator"""
        while self.population.generation == 0:
            if self.total_samples >= self.initial_sample_nums_max:
                print(f"\nInitialization reached max samples: {self.initial_sample_nums_max}")
                break

            prompt = self.problem.get_prompt_i1()
            self._sample_evaluate_register(prompt, "I1")

    def _iteratively_use_eohu_operator(self) -> None:
        """Evolution phase using E1, E2, M1, M2 operators"""
        while self._continue_loop():
            try:
                # E1: Explore different algorithm
                if self._continue_loop():
                    selected = [self.population.selection() for _ in range(self.selection_num)]
                    prompt = self.problem.get_prompt_e1(selected)
                    self._sample_evaluate_register(prompt, "E1")

                # E2: Exploit common insights
                if self._continue_loop():
                    selected = [self.population.selection() for _ in range(self.selection_num)]
                    prompt = self.problem.get_prompt_e2(selected)
                    self._sample_evaluate_register(prompt, "E2")

                # M1: Modify existing algorithm
                if self._continue_loop():
                    selected = self.population.selection()
                    prompt = self.problem.get_prompt_m1(selected)
                    self._sample_evaluate_register(prompt, "M1")

                # M2: Fine-tune parameters
                if self._continue_loop():
                    selected = self.population.selection()
                    prompt = self.problem.get_prompt_m2(selected)
                    self._sample_evaluate_register(prompt, "M2")

            except Exception as e:
                print(f"Evolution exception: {e}")
                continue

    def _multi_threaded_sampling(self, fn) -> None:
        """Execute sampling function with multiple threads"""
        with ThreadPoolExecutor(max_workers=self.num_samplers) as executor:
            futures = [executor.submit(fn) for _ in range(self.num_samplers)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Thread exception: {e}")

    def _update_evolution_summary(self) -> None:
        """Update the evolution summary file"""
        summary_file = os.path.join(self.base_path, 'evolution_summary.txt')

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"EoH-U Evolution Summary ({self.problem.name})\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total Samples: {self.total_samples}\n")
            f.write(f"Current Generation: {self.population.generation}\n")
            f.write(f"Global Best Fitness: {self.global_best_fitness:.4f}\n\n")

            if self.global_best_individual:
                f.write("="*80 + "\n")
                f.write("Global Best Individual:\n")
                f.write("="*80 + "\n")
                best = self.global_best_individual
                f.write(f"Fitness: {best.fitness:.4f}\n")
                f.write(f"Generation: {best.generation}\n")
                f.write(f"Sample ID: {best.sample_id}\n")
                f.write(f"Execution Time: {best.execution_time:.4f}s\n\n")

                f.write(f"Description:\n{best.thought}\n\n")

                f.write(f"Performance per instance:\n")
                for inst_name, fit in best.fitness_per_instance.items():
                    f.write(f"  {inst_name}: {fit:.4f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Generation-by-Generation Best Fitness:\n")
            f.write("="*80 + "\n")
            f.write("Gen\tBest Fitness\tDescription\n")
            f.write("-"*80 + "\n")

            # Collect best from each generation
            for gen in range(1, self.population.generation + 1):
                gen_dir = os.path.join(self.base_path, f'generation_{gen}')
                if os.path.exists(gen_dir):
                    best_file = os.path.join(gen_dir, 'individual_0_rank1.txt')
                    if os.path.exists(best_file):
                        with open(best_file, 'r', encoding='utf-8') as bf:
                            content = bf.read()
                            fitness_match = re.search(r'Fitness: ([\d.]+)', content)
                            desc_match = re.search(r'Algorithm Description:\n=+\n(.*?)\n', content, re.DOTALL)

                            if fitness_match:
                                fitness = float(fitness_match.group(1))
                                desc = desc_match.group(1).strip()[:50] + "..." if desc_match else "N/A"
                                f.write(f"{gen}\t{fitness:.4f}\t\t{desc}\n")

    def _save_final_results(self) -> None:
        """Save final evolution results"""
        # Save statistics
        stats = {
            'problem': self.problem.name,
            'total_samples': self.total_samples,
            'final_generation': self.population.generation,
            'global_best_fitness': self.global_best_fitness,
            'population_size': len(self.population),
        }

        with open(os.path.join(self.base_path, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        # Save best algorithm
        if self.global_best_individual:
            best = self.global_best_individual
            with open(os.path.join(self.base_path, 'best_algorithm.py'), 'w', encoding='utf-8') as f:
                f.write(f"# Best Algorithm (Fitness: {best.fitness:.2f})\n")
                f.write(f"# Problem: {self.problem.name}\n")
                f.write(f"# Description: {best.thought}\n\n")
                f.write(best.code)

            with open(os.path.join(self.base_path, 'best_algorithm.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Problem: {self.problem.name}\n")
                f.write(f"Fitness: {best.fitness:.2f}\n")
                f.write(f"Generation: {best.generation}\n")
                f.write(f"Sample ID: {best.sample_id}\n")
                f.write(f"Execution Time: {best.execution_time:.4f}s\n\n")
                f.write(f"Description:\n{best.thought}\n\n")
                f.write(f"Performance per instance:\n")
                for inst_name, fit in best.fitness_per_instance.items():
                    f.write(f"  {inst_name}: {fit:.2f}\n")

    def run(self) -> None:
        """Run complete EoH-U evolution"""
        start_time = time.time()

        print("\n" + "="*70)
        print(f"EoH-U Evolution Framework - {self.problem.name}")
        print("="*70)

        print("\n=== Training Set ===")
        self.problem.print_training_set_info()

        # Phase 1: Initialization
        print("\n" + "="*70)
        print("Phase 1: Initialization (I1 Operator)")
        print("="*70)

        self._multi_threaded_sampling(self._iteratively_init_population)

        # Force generation 1 if we have candidates
        self.population.force_generation_update()

        # Update global best
        best = self.population.get_best()
        if best and best.fitness < self.global_best_fitness:
            self.global_best_fitness = best.fitness
            self.global_best_individual = best
            self._update_evolution_summary()

        # Check if initialization succeeded
        if len(self.population) < self.selection_num:
            print(f"\nInitialization FAILED: Only {len(self.population)} valid individuals "
                  f"(need at least {self.selection_num})")
            self._save_final_results()
            return

        print(f"\nInitialization SUCCESS: {len(self.population)} valid individuals")
        print(f"  Best initial fitness: {self.population.get_best().fitness:.2f}")

        # Phase 2: Evolution
        print("\n" + "="*70)
        print("Phase 2: Evolution (E1, E2, M1, M2 Operators)")
        print("="*70)

        self._multi_threaded_sampling(self._iteratively_use_eohu_operator)

        # Save final results
        self._save_final_results()
        self._update_evolution_summary()

        elapsed_time = time.time() - start_time

        # Print final summary
        print("\n" + "="*70)
        print("EoH-U Evolution Complete")
        print("="*70)
        print(f"Problem: {self.problem.name}")
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        print(f"Total samples: {self.total_samples}")
        print(f"Final generation: {self.population.generation}")

        if self.global_best_individual:
            print(f"\nBest Algorithm Found:")
            print(f"  Fitness: {self.global_best_fitness:.2f}")
            print(f"  Description: {self.global_best_individual.thought}")
            print(f"\n  Performance per instance:")
            for inst_name, fit in self.global_best_individual.fitness_per_instance.items():
                print(f"    {inst_name}: {fit:.2f}")
        else:
            print("\nNo valid solution found")

        print(f"\nAll results saved to: {self.base_path}")
        print(f"  - evolution_summary.txt: Overall summary")
        print(f"  - generation_X/: Per-generation details")
        print("="*70)
