"""
Main entry point for EoH-U framework
Supports both TSP and CVRP problems
"""

import os
import time
import argparse
from openai import OpenAI

from core import EoHU_Evolution
from problems.tsp import TSPProblem
from problems.cvrp import CVRPProblem
from config import TSP_CONFIG, CVRP_CONFIG

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='EoH-U: Evolution of Heuristics Framework')
    parser.add_argument('--problem', type=str, default='tsp', choices=['tsp', 'cvrp'],
                       help='Problem type to solve (default: tsp)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file (optional)')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        # Load custom config
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Load default config
        if args.problem.lower() == 'tsp':
            config = TSP_CONFIG
        elif args.problem.lower() == 'cvrp':
            config = CVRP_CONFIG
        else:
            raise ValueError(f"Unknown problem type: {args.problem}")

    print("\n" + "="*70)
    print(f"EoH-U Framework - {config['problem_name']}")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Problem: {config['problem_name']}")
    print(f"  Population Size: {config['pop_size']}")
    print(f"  Max Generations: {config['max_generations']}")
    print(f"  Max Samples: {config['max_sample_nums']}")
    print(f"  Timeout: {config['timeout']}s")
    print(f"  Model: {config['model']}")
    print("="*70)

    # Initialize OpenAI client
    client = OpenAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )

    # Initialize problem
    print(f"\nInitializing {config['problem_name']} problem...")
    if args.problem.lower() == 'tsp':
        problem = TSPProblem(seed=config['seed'])
    elif args.problem.lower() == 'cvrp':
        problem = CVRPProblem(seed=config['seed'])
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

    # Create output directory
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    base_path = f"{config['base_path_prefix']}_{config['model']}_{timestamp}"
    os.makedirs(base_path, exist_ok=True)

    # Initialize evolution framework
    print(f"Initializing evolution framework...")
    evolution = EoHU_Evolution(
        llm_client=client,
        llm_model=config['model'],
        problem=problem,
        pop_size=config['pop_size'],
        max_generations=config['max_generations'],
        max_sample_nums=config['max_sample_nums'],
        selection_num=config['selection_num'],
        num_samplers=config['num_samplers'],
        base_path=base_path,
        timeout=config['timeout']
    )

    # Run evolution
    print(f"\nStarting evolution...")
    evolution.run()

    print(f"\nResults saved to: {base_path}")


if __name__ == '__main__':
    main()
