# main.py

import time
import json
import pandas as pd
import itertools
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tsp_ga import GeneticAlgorithm
from problem_utils import load_tsplib_problem
from utils import (
    compute_per_run_stats,
    save_per_run_statistics,
    prepare_per_generation_stats,
    save_per_generation_statistics,
    average_per_run_stats,
    average_per_gen_stats
)

N_RUNS_PER_EXPERIMENT = 3

def run_experiment(args):
    problem_file, configuration = args
    (selection_method, crossover_method, mutation_method, elitism, diversity_injection) = configuration

    # Load optimal fitness
    with open('problems/optimal_solutions.json') as f:
        optimal_solutions = json.load(f)
    optimal_fitness = optimal_solutions[problem_file]

    # Load problem
    problem = load_tsplib_problem('problems/' + problem_file + '.tsp')

    per_run_stats_list = []
    per_gen_stats_list = []

    for run in range(N_RUNS_PER_EXPERIMENT):
        ga_params = {
            'population_size'   : 75,
            'mutation_rate'     : 0.1,
            'crossover_rate'    : 0.8,
            'generations'       : 3000,
            'selection_method'  : selection_method,
            'crossover_method'  : crossover_method,
            'mutation_method'   : mutation_method,
            'elitism'           : elitism,
            'diversity_injection': diversity_injection,
            'patience'          : 200,
            'optimal_fitness'   : optimal_fitness,
            'print_info'        : False
        }

        ga = GeneticAlgorithm(
            problem=problem,
            **ga_params
        )

        start_time = time.time()
        best_solution, stats = ga.run()
        time_taken = time.time() - start_time

        # Collect per-run statistics
        per_run_stats = compute_per_run_stats(
            problem_file=problem_file,
            ga=ga,
            best_solution=best_solution,
            stats=stats,
            time_taken=time_taken,
            optimal_fitness=optimal_fitness
        )
        per_run_stats_list.append(per_run_stats)

        # Collect per-generation statistics
        per_gen_stats = prepare_per_generation_stats(
            problem_file=problem_file,
            ga=ga,
            stats=stats,
            optimal_fitness=optimal_fitness
        )
        per_gen_stats_list.append(per_gen_stats)

    # Average per-run statistics
    avg_per_run_stats = average_per_run_stats(per_run_stats_list)

    # Average per-generation statistics
    avg_per_gen_stats = average_per_gen_stats(per_gen_stats_list)

    return (avg_per_run_stats, avg_per_gen_stats)

def main():
    # Problem instances
    small_instances     = ['ulysses22', 'gr48', 'berlin52', 'brazil58', 'st70']
    medium_instances    = ['eil101', 'lin105', 'gr137', 'kroA150', 'si175']
    large_instances     = ['gr202', 'kroA200', 'ts225', 'pr226', 'a280']

    all_instances = small_instances + medium_instances + large_instances

    # Parameter options
    selection_methods           = ['tournament', 'sus', 'rank']
    crossover_methods           = ['pmx', 'order']
    mutation_methods            = ['swap', 'inversion', 'scramble']
    elitism_options             = [True, False]
    diversity_injection_options = [True, False]

    # Create folder for results
    os.makedirs("results", exist_ok=True)
    
    # Load existing CSV files if they exist
    per_run_csv_file        = 'results/ga_results.csv'
    per_generation_csv_file = 'results/ga_per_generation_results.csv'

    try:
        per_run_df = pd.read_csv(per_run_csv_file)
    except FileNotFoundError:
        per_run_df = pd.DataFrame()

    # Load optimal fitness
    with open('problems/optimal_solutions.json') as f:
        optimal_solutions = json.load(f)

    # Generate all configurations
    configurations = list(itertools.product(
        selection_methods,
        crossover_methods,
        mutation_methods,
        elitism_options,
        diversity_injection_options
    ))

    # Build the list of experiments to run
    experiments = []
    for problem_file in all_instances:
        for configuration in configurations:
            (selection_method, crossover_method, mutation_method, elitism, diversity_injection) = configuration

            if not per_run_df.empty:
                matching_rows = per_run_df[
                    (per_run_df['problem_file'] == problem_file) &
                    (per_run_df['selection_method'] == selection_method) &
                    (per_run_df['crossover_method'] == crossover_method) &
                    (per_run_df['mutation_method'] == mutation_method) &
                    (per_run_df['elitism'] == int(elitism)) &
                    (per_run_df['diversity_injection'] == int(diversity_injection))
                ]
                if not matching_rows.empty:
                    continue  # Skip this experiment

            experiments.append((problem_file, configuration))

    total_experiments = len(experiments)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_experiment, experiment): experiment for experiment in experiments}

        for future in tqdm(as_completed(futures), total=total_experiments, desc='Experiments'):
            experiment = futures[future]
            try:
                avg_per_run_stats, avg_per_gen_stats = future.result()
                # Save the results to CSV files
                save_per_run_statistics(avg_per_run_stats, per_run_csv_file)
                save_per_generation_statistics(avg_per_gen_stats, per_generation_csv_file)
            except Exception as e:
                print(f'Experiment {experiment} generated an exception: {e}')

if __name__ == '__main__':
    main()
