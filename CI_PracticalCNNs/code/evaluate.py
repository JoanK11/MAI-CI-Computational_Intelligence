import os
import csv
import torch
from sklearn.metrics import f1_score

def ensure_results_dir():
    if not os.path.exists('results'):
        os.makedirs('results')

def write_results_to_csv(results, filename):
    ensure_results_dir()
    filepath = os.path.join('results', filename)
    if not results:
        print(f"No results to write to {filepath}.")
        return
    fieldnames = results[0].keys()
    
    with open(filepath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:
            writer.writeheader()
        for result in results:
            writer.writerow(result)
            
def read_tuning_results_from_csv(filename):
    filepath = os.path.join('results', filename)
    if not os.path.exists(filepath):
        print(f"Tuning results file {filepath} does not exist.")
        return []
    
    results = []
    with open(filepath, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                row_converted = {
                    'learning_rate': float(row['learning_rate']),
                    'weight_decay': float(row['weight_decay']),
                    'epochs': int(row['epochs']),
                    'betas': eval(row['betas']),
                    'ol_activation': row['ol_activation'] if row['ol_activation'] != 'None' else None,
                    'cost_function': row['cost_function'],
                    'validation_accuracy': float(row['validation_accuracy']),
                    'validation_f1_score': float(row['validation_f1_score'])
                }
                results.append(row_converted)
            except Exception as e:
                print(f"Error parsing row: {row}")
                print(f"Exception: {e}")
    return results

def evaluate_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    # print(f'Eval Accuracy: {accuracy:.2f}%, F1-Score: {f1:.2f}')
    
    return accuracy, f1
