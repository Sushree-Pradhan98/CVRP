# ✅ Parameter Tuning Script for CVRP Metaheuristics
import os
import pandas as pd
import numpy as np
from Genetic_Algorithm import GeneticAlgorithmCVRP, load_cvrp_data as load_ga
from Simulated_Annealing import SimulatedAnnealingCVRP, load_cvrp_data as load_sa
from Tabu_Search import TabuSearchCVRP, load_cvrp_data as load_tabu
from Random_Search import RandomCVRP, load_cvrp_data as load_random
from Greedy_Algorithm import GreedyCVRP, load_cvrp_data as load_greedy

def run_tuning(file_path, algorithm, param_grid, runs=10):
    """
    Runs tuning experiments for different CVRP solvers.

    :param file_path: Path to the VRP data file.
    :param algorithm: The metaheuristic algorithm being tuned ('ga', 'sa', 'tabu', 'random', 'greedy').
    :param param_grid: List of parameter dictionaries to test.
    :param runs: Number of times each parameter setting is evaluated.
    :return: Pandas DataFrame containing tuning results.
    """
    # Load data according to the selected algorithm
    if algorithm in ['ga', 'sa', 'tabu']:
        loc, dem, cap = load_ga(file_path)
    elif algorithm == 'random':
        loc, dem, cap = load_random(file_path)
    elif algorithm == 'greedy':
        loc, dem, cap = load_greedy(file_path)

    results = []

    for params in param_grid:
        distances = []

        for _ in range(runs):
            if algorithm == 'ga':
                solver = GeneticAlgorithmCVRP(
                    loc, dem, cap,
                    population_size=params['population_size'],
                    mutation_rate=params['mutation_rate'],
                    max_evaluations=params['max_evaluations']
                )
                solution = solver.run()
                dist = solver.total_distance(solution)

            elif algorithm == 'sa':
                solver = SimulatedAnnealingCVRP(
                    loc, dem, cap,
                    initial_temp=params['initial_temp'],
                    cooling_rate=params['cooling_rate'],
                    max_evaluations=params['max_evaluations']
                )
                _, dist = solver.simulated_annealing()

            elif algorithm == 'tabu':
                solver = TabuSearchCVRP(
                    loc, dem, cap,
                    tabu_tenure=params['tabu_tenure'],
                    max_evaluations=params['max_evaluations']
                )
                _, dist = solver.tabu_search()

            elif algorithm == 'random':
                solver = RandomCVRP(
                    loc, dem, cap,
                    max_evaluations=params['max_evaluations']
                )
                solution = solver.run()
                dist = solver.total_distance(solution)

            elif algorithm == 'greedy':
                solver = GreedyCVRP(loc, dem, cap)
                solution = solver.greedy_solution()
                dist = solver.total_distance(solution)

            distances.append(dist)

        results.append({
            'params': params,
            'best': round(np.min(distances), 1),
            'worst': round(np.max(distances), 1),
            'avg': round(np.mean(distances), 1),
            'std': round(np.std(distances), 1)
        })

    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    vrp_file = "Data/A-n32-k5.vrp"

    ga_params = [
        {'population_size': 10, 'mutation_rate': 0.05, 'max_evaluations': 10000},
        {'population_size': 20, 'mutation_rate': 0.1, 'max_evaluations': 20000},
        {'population_size': 30, 'mutation_rate': 0.2, 'max_evaluations': 30000},
    ]

    sa_params = [
        {'initial_temp': 1000, 'cooling_rate': 0.99, 'max_evaluations': 10000},
        {'initial_temp': 500, 'cooling_rate': 0.98, 'max_evaluations': 20000},
        {'initial_temp': 1500, 'cooling_rate': 0.95, 'max_evaluations': 30000},
    ]

    tabu_params = [
        {'tabu_tenure': 10, 'max_evaluations': 10000},
        {'tabu_tenure': 20, 'max_evaluations': 20000},
        {'tabu_tenure': 30, 'max_evaluations': 30000},
    ]

    random_params = [
        {'max_evaluations': 10000},
        {'max_evaluations': 20000},
        {'max_evaluations': 30000},
    ]

    print("\n--- Genetic Algorithm Tuning ---")
    ga_results = run_tuning(vrp_file, 'ga', ga_params)
    print(ga_results)
    ga_results.to_csv("results_ga.csv", index=False)

    print("\n--- Simulated Annealing Tuning ---")
    sa_results = run_tuning(vrp_file, 'sa', sa_params)
    print(sa_results)
    sa_results.to_csv("results_sa.csv", index=False)

    print("\n--- Tabu Search Tuning ---")
    tabu_results = run_tuning(vrp_file, 'tabu', tabu_params)
    print(tabu_results)
    tabu_results.to_csv("results_tabu.csv", index=False)

    print("\n--- Random Search Tuning ---")
    random_results = run_tuning(vrp_file, 'random', random_params)
    print(random_results)
    random_results.to_csv("results_random.csv", index=False)
