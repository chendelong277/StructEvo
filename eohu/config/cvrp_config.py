"""
Configuration for CVRP problem
"""

CVRP_CONFIG = {
    # Problem settings
    'problem_name': 'CVRP',
    'seed': 42,

    # Evolution parameters
    'pop_size': 4,
    'max_generations': 10,
    'max_sample_nums': 50,
    'selection_num': 2,
    'num_samplers': 4,
    'timeout': 120,  # seconds (CVRP needs more time)

    # API configuration
    'api_key': 'YOUR_API_KEY',
    'base_url': 'YOUR_BASE_URL',
    'model': 'deepseek-v3',

    # Output settings
    'base_path_prefix': 'run/eohu_cvrp',
}
