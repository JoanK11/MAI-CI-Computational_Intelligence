# utils.py
import csv

def compute_per_run_stats(problem_file, ga, best_solution, stats, time_taken, optimal_fitness=None):
    """
    Computes per-run statistics.

    Args:
        problem_file (str): Path to the problem file.
        ga (GeneticAlgorithm): The GA instance.
        best_solution (Individual): The best solution found.
        stats (dict): Statistics collected during the run.
        time_taken (float): Total time taken for the run.
        optimal_fitness (float): The optimal fitness value for the problem.

    Returns:
        dict: A dictionary containing per-run statistics.
    """
    # Compute convergence rate
    convergence_rate = stats['generations_run'] / ga.generations

    # Collect experimental parameters (features)
    # Only include specified parameters
    parameters = {
        'problem_file': problem_file,
        'selection_method': ga.selection_method,
        'crossover_method': ga.crossover_method,
        'mutation_method': ga.mutation_method,
        'elitism': int(ga.elitism),
        'diversity_injection': int(ga.diversity_injection),
    }

    # Compute final best fitness error percentage
    if optimal_fitness is not None:
        final_best_error = ((best_solution.fitness - optimal_fitness) / optimal_fitness) * 100
    else:
        final_best_error = best_solution.fitness  # Use raw fitness if optimal not provided

    # Collect per-run data (results)
    per_run_data = {
        'final_best_fitness_error': final_best_error,
        'convergence_rate': convergence_rate,
        'time_taken': time_taken,
    }

    # Combine parameters and per-run data
    data_row = {**parameters, **per_run_data}

    return data_row  # Return for further use if needed

def save_per_run_statistics(data_row, csv_file):
    """
    Saves per-run statistics to a CSV file.

    Args:
        data_row (dict): Data to write to the CSV.
        csv_file (str): Filename of the CSV file.
    """
    # Only include specified parameters and results in the CSV columns
    csv_columns = [
        'problem_file',
        'selection_method',
        'crossover_method',
        'mutation_method',
        'elitism',
        'diversity_injection',
        'final_best_fitness_error',
        'convergence_rate',
        'time_taken',
    ]

    try:
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            # Write header if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(data_row)
    except IOError:
        print("I/O error when writing per-run statistics.")

def prepare_per_generation_stats(problem_file, ga, stats, optimal_fitness=None):
    """
    Prepares per-generation statistics.

    Args:
        problem_file (str): Path to the problem file.
        ga (GeneticAlgorithm): The GA instance.
        stats (dict): Statistics collected during the run.
        optimal_fitness (float): The optimal fitness value for the problem.

    Returns:
        list: A list of dictionaries containing per-generation statistics.
    """
    # Collect experimental parameters (features)
    # Only include specified parameters
    parameters = {
        'problem_file': problem_file,
        'selection_method': ga.selection_method,
        'crossover_method': ga.crossover_method,
        'mutation_method': ga.mutation_method,
        'elitism': int(ga.elitism),
        'diversity_injection': int(ga.diversity_injection),
    }

    # Prepare per-generation data
    per_gen_data = []
    for idx, (best_error, avg_error, diversity_hamming, diversity_matrix) in enumerate(zip(
            stats['best_fitness_per_gen'], stats['avg_fitness_per_gen'],
            stats['diversity_hamming_per_gen'], stats['diversity_matrix_per_gen'])):
        gen_number = idx * 50  # Since we record every 50 generations
        row = {
            **parameters,  # Include specified parameters
            'generation'        : gen_number,
            'best_fitness_error': best_error,
            'avg_fitness_error' : avg_error,
            'diversity_hamming' : diversity_hamming,
            'diversity_matrix'  : diversity_matrix
        }
        per_gen_data.append(row)
    return per_gen_data

def save_per_generation_statistics(per_gen_data, csv_file):
    """
    Saves per-generation statistics to a CSV file.

    Args:
        per_gen_data (list): List of data rows to write to the CSV.
        csv_file (str): Filename of the CSV file.
    """
    # Only include specified parameters and results in the CSV columns
    csv_columns = [
        'problem_file',
        'selection_method',
        'crossover_method',
        'mutation_method',
        'elitism',
        'diversity_injection',
        'generation',
        'best_fitness_error',
        'avg_fitness_error',
        'diversity_hamming',
        'diversity_matrix'
    ]

    try:
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            # Write header if file is empty
            if csvfile.tell() == 0:
                writer.writeheader()
            for row in per_gen_data:
                writer.writerow(row)
    except IOError:
        print("I/O error when writing per-generation statistics.")

def print_final_results(best_solution, time_taken, convergence_rate, optimal_fitness=None):
    """
    Prints the best solution found and runtime information.

    Args:
        best_solution (Individual): The best solution found.
        time_taken (float): Total time taken for the run.
        convergence_rate (float): Convergence rate as a fraction.
        optimal_fitness (float): The optimal fitness value for the problem.
    """
    print("-------------------------------------------")
    if optimal_fitness is not None:
        final_error = ((best_solution.fitness - optimal_fitness) / optimal_fitness) * 100
        print(f"Best tour length error: {final_error:.2f}%")
    else:
        print("Best tour length:", best_solution.fitness)
    print("Best tour:", best_solution.chromosome)
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Convergence rate: {convergence_rate * 100:.2f}%")
    print("-------------------------------------------")


def average_per_run_stats(per_run_stats_list):
    """
    Averages the per-run statistics over multiple runs.

    Returns a single dict with averaged per-run statistics.
    """
    # All per_run_stats in the list have the same parameters
    # We need to average 'final_best_fitness_error', 'convergence_rate', 'time_taken'

    # Copy parameters from the first run
    avg_stats = per_run_stats_list[0].copy()

    # Initialize sums
    total_final_best_fitness_error = 0
    total_convergence_rate = 0
    total_time_taken = 0

    for stats in per_run_stats_list:
        total_final_best_fitness_error += stats['final_best_fitness_error']
        total_convergence_rate += stats['convergence_rate']
        total_time_taken += stats['time_taken']

    num_runs = len(per_run_stats_list)

    # Compute averages
    avg_stats['final_best_fitness_error'] = total_final_best_fitness_error / num_runs
    avg_stats['convergence_rate'] = total_convergence_rate / num_runs
    avg_stats['time_taken'] = total_time_taken / num_runs

    return avg_stats

def average_per_gen_stats(per_gen_stats_list):
    """
    Averages the per-generation statistics over multiple runs.

    Returns a list of dicts with averaged per-generation statistics.
    """
    # per_gen_stats_list is a list of lists of dicts
    # Each list corresponds to one run
    # Each dict in the list corresponds to stats at a generation in that run

    # Build a dictionary to hold lists of stats for each generation
    gen_stats_dict = {}

    # Collect stats from all runs
    for run_stats in per_gen_stats_list:
        for gen_stats in run_stats:
            gen_number = gen_stats['generation']
            if gen_number not in gen_stats_dict:
                gen_stats_dict[gen_number] = []
            gen_stats_dict[gen_number].append(gen_stats)

    # Now, for each generation, average the stats over the runs that have data for that generation
    avg_per_gen_data = []

    for gen_number in sorted(gen_stats_dict.keys()):
        gen_stats_list = gen_stats_dict[gen_number]

        # Initialize sums
        total_best_fitness_error = 0
        total_avg_fitness_error = 0
        total_diversity_hamming = 0
        total_diversity_matrix  = 0

        num_runs = len(gen_stats_list)

        for gen_stats in gen_stats_list:
            total_best_fitness_error += gen_stats['best_fitness_error']
            total_avg_fitness_error += gen_stats['avg_fitness_error']
            total_diversity_hamming += gen_stats['diversity_hamming']
            total_diversity_matrix  += gen_stats['diversity_matrix']

        # Average stats for this generation
        avg_gen_stats = gen_stats_list[0].copy()  # Copy parameters
        avg_gen_stats['best_fitness_error'] = total_best_fitness_error / num_runs
        avg_gen_stats['avg_fitness_error']  = total_avg_fitness_error / num_runs
        avg_gen_stats['diversity_hamming']  = total_diversity_hamming / num_runs
        avg_gen_stats['diversity_matrix']   = total_diversity_matrix / num_runs

        avg_per_gen_data.append(avg_gen_stats)

    return avg_per_gen_data
