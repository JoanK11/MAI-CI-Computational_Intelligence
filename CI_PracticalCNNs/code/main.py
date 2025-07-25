# main.py
import os
import csv
import torch
from model import CNN
from data_loader import load_data
from train import train_model, hyperparameter_tuning
from evaluate import evaluate_model, write_results_to_csv
from tqdm import tqdm
import statistics

def read_results_from_csv(filename):
    filepath = os.path.join('results', filename)
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        return []
    
    results = []
    with open(filepath, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:

            row_converted = {
                'num_blocks': int(row['num_blocks']),
                'initial_filters': int(row['initial_filters']),
                'nhl_activation': row['nhl_activation'],
                'data_split': eval(row['data_split']),
            }
            results.append(row_converted)
    return results



def read_experiment_results(filename):
    filepath = os.path.join('results', filename)
    if not os.path.exists(filepath):
        print(f"Experiment results file {filepath} does not exist.")
        return set()
    
    existing_configs = set()
    with open(filepath, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                config = (
                    int(row['num_blocks']),
                    int(row['initial_filters']),
                    row['nhl_activation'],
                    eval(row['data_split'])
                )
                existing_configs.add(config)
            except Exception as e:
                print(f"Error parsing row: {row}")
                print(f"Exception: {e}")
    return existing_configs

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    best_params     = hyperparameter_tuning()
    best_lr         = best_params['learning_rate']
    best_wd         = best_params['weight_decay']
    best_epochs     = best_params['epochs']
    best_betas      = best_params['betas']
    best_criterion  = best_params['cost_function']
    best_ol_activation = best_params['ol_activation']
    
    print("\nStarting main experiments with tuned hyperparameters...")
    
    base_splits             = [(0.8, 0.1, 0.1), (0.4, 0.2, 0.4), (0.1, 0.1, 0.8)]
    initial_filters_list    = [2, 4, 8, 16, 32, 64, 128]
    nhl_activation_list     = ['relu', 'sigmoid']
    num_blocks_list         = [1, 2, 3, 4] # 1 to 4 (5 is too small!)
    
    # we load existing experiment configurations to skip them
    experiment_filename = 'experiment_results.csv'
    existing_configs = read_experiment_results(experiment_filename)
    print(f"Loaded {len(existing_configs)} existing experiment configurations from '{experiment_filename}'.")
    
    results = []
    
    experiment_params = []
    for initial_filters in initial_filters_list:
        for num_blocks in num_blocks_list:
            for nhl_activation in nhl_activation_list:
                for split in base_splits:
                    config = (num_blocks, initial_filters, nhl_activation, split)
                    if config in existing_configs:
                        print(f"Skipping already tested configuration: Filters={initial_filters}, Blocks={num_blocks}, Activation={nhl_activation}, Split={split}")
                        continue
                    experiment_params.append(config)
    
    total_new_experiments = len(experiment_params) * 3  # 3 runs each
    print(f"Total new main experiments to run (including 3 runs each): {total_new_experiments}")
    
    with tqdm(total=total_new_experiments, desc="Main Experiments") as pbar:
        for num_blocks, initial_filters, nhl_activation, split in experiment_params:
            try:
                test_accuracies = []
                f1_scores = []
                
                for run in range(1, 4):
                    try:
                        train_loader, val_loader, test_loader = load_data(batch_size=32, train_val_test_split=split)
        
                        model = CNN(num_blocks=num_blocks, initial_filters=initial_filters, nhl_activation=nhl_activation, ol_activation=best_ol_activation)
        
                        trained_model = train_model(model, train_loader, val_loader, learning_rate=best_lr, weight_decay=best_wd, betas=best_betas, epochs=best_epochs, criterion_name=best_criterion, device=device)
                        test_accuracy, f1 = evaluate_model(trained_model, test_loader, device=device)
                        
                        test_accuracies.append(test_accuracy)
                        f1_scores.append(f1)
                        
                        # print(f"Run {run}/3 for Config (Filters: {initial_filters}, Blocks: {num_blocks}, Activation: {nhl_activation}, Split: {split}) - Accuracy: {test_accuracy:.4f}, F1: {f1:.4f}")
                    except Exception as e_run:
                        print(f"Exception during run {run} for experiment with Initial Filters: {initial_filters}, Blocks: {num_blocks}, NHL Activation: {nhl_activation}, Split: {split}")
                        print(f"Error: {e_run}")
                    
                    pbar.update(1)
                
                if test_accuracies and f1_scores:
                    avg_accuracy = statistics.mean(test_accuracies)
                    avg_f1 = statistics.mean(f1_scores)
                    std_accuracy = statistics.stdev(test_accuracies) if len(test_accuracies) > 1 else 0.0
                    std_f1 = statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0
                    
                    results.append({
                        'num_blocks': num_blocks,
                        'initial_filters': initial_filters,
                        'filter_sizes': [initial_filters * (2 ** i) for i in range(num_blocks)],
                        'nhl_activation': nhl_activation,
                        'ol_activation': best_ol_activation,
                        'cost_function': best_criterion,
                        'data_split': split,
                        'learning_rate': best_lr,
                        'weight_decay': best_wd,
                        'betas': best_betas,
                        'epochs': best_epochs,
                        'average_test_accuracy': avg_accuracy,
                        'std_test_accuracy': std_accuracy,
                        'average_f1_score': avg_f1,
                        'std_f1_score': std_f1
                    })
                else:
                    print(f"No successful runs for Config (Filters: {initial_filters}, Blocks: {num_blocks}, Activation: {nhl_activation}, Split: {split}). Skipping.")
            except ValueError as e:
                print(f"Skipping configuration due to {e}")
            except Exception as e:
                print(f"Exception during experiment with Initial Filters: {initial_filters}, Blocks: {num_blocks}, NHL Activation: {nhl_activation}, Split: {split}")
                print(f"Error: {e}")

    # Save new experiment results
    write_results_to_csv(results, filename=experiment_filename)
    print(f"\nMain experiments completed. Results saved to 'results/{experiment_filename}'.")

if __name__ == '__main__':
    main()
