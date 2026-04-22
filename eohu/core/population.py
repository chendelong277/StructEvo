"""
Population class: Manages the population of algorithm individuals
"""

import os
import numpy as np
from typing import List, Optional
import random

from .individual import Individual


class Population:
    """EoH-style population with registration and survival mechanisms"""

    def __init__(self, pop_size: int, base_path: Optional[str] = None):
        """
        Initialize population

        Args:
            pop_size: Maximum population size
            base_path: Base directory for saving generation data
        """
        self.pop_size = pop_size
        self._pop: List[Individual] = []
        self._next_gen_pop: List[Individual] = []
        self.generation = 0
        self.base_path = base_path

    def register_function(self, individual: Individual) -> None:
        """
        Register individual to candidate pool
        Only successful individuals (fitness != inf) are registered

        Args:
            individual: Individual to register
        """
        if individual.is_valid():
            self._next_gen_pop.append(individual)
            print(f"  Valid individual registered (fitness={individual.fitness:.2f}, "
                  f"candidates: {len(self._next_gen_pop)}/{self.pop_size})")

            # Execute survival when pool reaches pop_size
            if len(self._next_gen_pop) >= self.pop_size:
                self.survival()
        else:
            print(f"  Invalid individual, not registered")

    def survival(self) -> None:
        """
        Select best pop_size individuals from current population + candidate pool
        Save generation data before updating
        """
        combined = self._pop + self._next_gen_pop
        combined.sort(key=lambda x: x.fitness)
        self._pop = combined[:self.pop_size]
        self._next_gen_pop = []
        self.generation += 1

        print(f"\n{'='*60}")
        print(f"Generation {self.generation} Update")
        print(f"  Best fitness: {self._pop[0].fitness:.2f}")
        print(f"  Population size: {len(self._pop)}")
        print(f"{'='*60}\n")

        # Save this generation's data
        if self.base_path:
            self._save_generation()

    def _save_generation(self) -> None:
        """Save current generation's population to disk"""
        if not self._pop:
            return

        gen_dir = os.path.join(self.base_path, f'generation_{self.generation}')
        os.makedirs(gen_dir, exist_ok=True)

        # Save each individual
        for i, ind in enumerate(self._pop):
            # Save detailed info
            with open(os.path.join(gen_dir, f'individual_{i}_rank{i+1}.txt'), 'w', encoding='utf-8') as f:
                f.write(f"="*70 + "\n")
                f.write(f"Generation {self.generation} - Individual {i} (Rank {i+1})\n")
                f.write(f"="*70 + "\n\n")

                f.write(f"Fitness: {ind.fitness:.4f}\n")
                f.write(f"Execution Time: {ind.execution_time:.4f}s\n")
                f.write(f"Sample ID: {ind.sample_id}\n\n")

                f.write(f"Performance per instance:\n")
                for inst_name, fit in ind.fitness_per_instance.items():
                    fitness_str = f"{fit:.4f}" if fit != float('inf') else "Failed"
                    f.write(f"  {inst_name}: {fitness_str}\n")

                f.write(f"\n" + "="*70 + "\n")
                f.write(f"Algorithm Description:\n")
                f.write(f"="*70 + "\n")
                f.write(f"{ind.thought}\n\n")

                f.write(f"="*70 + "\n")
                f.write(f"Algorithm Code:\n")
                f.write(f"="*70 + "\n")
                f.write(f"{ind.code}\n")

            # Save just the code separately for easy testing
            with open(os.path.join(gen_dir, f'individual_{i}_code.py'), 'w', encoding='utf-8') as f:
                f.write(f"# Generation {self.generation} - Rank {i+1}\n")
                f.write(f"# Fitness: {ind.fitness:.4f}\n")
                f.write(f"# Description: {ind.thought}\n\n")
                f.write(ind.code)

        # Save generation summary
        with open(os.path.join(gen_dir, 'generation_summary.txt'), 'w', encoding='utf-8') as f:
            f.write(f"="*70 + "\n")
            f.write(f"Generation {self.generation} Summary\n")
            f.write(f"="*70 + "\n\n")

            f.write(f"Population Size: {len(self._pop)}\n")
            f.write(f"Best Fitness: {self._pop[0].fitness:.4f}\n")
            f.write(f"Worst Fitness: {self._pop[-1].fitness:.4f}\n")

            fitnesses = [ind.fitness for ind in self._pop]
            f.write(f"Average Fitness: {np.mean(fitnesses):.4f}\n")
            f.write(f"Std Dev: {np.std(fitnesses):.4f}\n\n")

            f.write(f"="*70 + "\n")
            f.write(f"Best Individual (Rank 1):\n")
            f.write(f"="*70 + "\n")
            best = self._pop[0]
            f.write(f"Fitness: {best.fitness:.4f}\n")
            f.write(f"Description: {best.thought}\n\n")
            f.write(f"Performance per instance:\n")
            for inst_name, fit in best.fitness_per_instance.items():
                f.write(f"  {inst_name}: {fit:.4f}\n")

            f.write(f"\n" + "="*70 + "\n")
            f.write(f"Population Ranking:\n")
            f.write(f"="*70 + "\n")
            f.write(f"Rank\tFitness\t\tDescription\n")
            f.write(f"-"*70 + "\n")
            for i, ind in enumerate(self._pop):
                desc_short = ind.thought[:50] + "..." if len(ind.thought) > 50 else ind.thought
                f.write(f"{i+1}\t{ind.fitness:.4f}\t\t{desc_short}\n")

        print(f"  Generation {self.generation} saved to {gen_dir}")

    def selection(self) -> Individual:
        """Tournament selection"""
        tournament_size = min(3, len(self._pop))
        candidates = random.sample(self._pop, tournament_size)
        return min(candidates, key=lambda x: x.fitness)

    def get_best(self) -> Optional[Individual]:
        """Get the best individual in current population"""
        if not self._pop:
            return None
        return self._pop[0]

    def force_generation_update(self) -> None:
        """Force update to generation 1 if we have candidates but generation is still 0"""
        if self.generation == 0 and len(self._next_gen_pop) > 0:
            print(f"\nForcing Generation 1 with {len(self._next_gen_pop)} valid individuals")
            self._next_gen_pop.sort(key=lambda x: x.fitness)
            self._pop = self._next_gen_pop[:min(self.pop_size, len(self._next_gen_pop))]
            self._next_gen_pop = []
            self.generation = 1

            # Save generation 1
            if self.base_path:
                self._save_generation()

    def __len__(self):
        return len(self._pop)
