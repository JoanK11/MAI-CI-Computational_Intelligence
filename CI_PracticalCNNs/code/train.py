# train.py
import torch
import torch.optim as optim
from torch import nn

import os

from model import CNN
from data_loader import load_data
from evaluate import (
    evaluate_model, 
    write_results_to_csv, 
    read_tuning_results_from_csv
)
from tqdm import tqdm
import itertools



def hyperparameter_tuning():
    tuning_filename = 'tuning_results.csv'
    tuning_filepath = os.path.join('results', tuning_filename)
    
    # check if tuning results already exist
    if os.path.exists(tuning_filepath):
        print(f"Found existing hyperparameter tuning results at {tuning_filepath}. Loading results...")
        tuning_results = read_tuning_results_from_csv(tuning_filename)
        if not tuning_results:
            print("Existing tuning results file is empty. Proceeding with hyperparameter tuning.")
        else:
            # Find the best hyperparameters based on validation accuracy
            best_result = max(tuning_results, key=lambda x: x['validation_accuracy'])
            print(f"Best hyperparameters from existing results: {best_result}")
            return best_result
    
    print("No existing hyperparameter tuning results found. Starting hyperparameter tuning...")
    
    learning_rates  = [0.01, 0.001, 0.0001]
    weight_decays   = [0, 0.0001, 0.001]
    epochs_list     = [30, 50]
    betas_list      = [(0.9, 0.999), (0.5, 0.999), (0.9, 0.9)]
    ol_activations  = [None, 'softmax']
    cost_functions  = ['cross_entropy', 'nll_loss']
    
    device          = 'cuda' if torch.cuda.is_available() else 'cpu'
    nhl_activation  = 'relu' # fixed for tunning
    initial_filters = 32     #       "
    num_blocks      = 3      #       "

    # fixed split (80, 10, 10)
    train_loader, val_loader, _ = load_data()
    tuning_results = []

    tuning_combinations = list(itertools.product(
        learning_rates,
        weight_decays,
        epochs_list,
        betas_list,
        ol_activations,
        cost_functions
    ))

    print("Starting hyperparameter tuning...")
    for (lr, wd, epochs, betas, ol_activation, criterion_name) in tqdm(tuning_combinations, desc="Hyperparameter Tuning"):
        try:
            model = CNN(num_blocks=num_blocks, initial_filters=initial_filters, nhl_activation=nhl_activation, ol_activation=ol_activation)
            trained_model = train_model(model, train_loader, val_loader, learning_rate=lr, weight_decay=wd, betas=betas, epochs=epochs, criterion_name=criterion_name, device=device)

            # evaluate on validation set
            val_accuracy, val_f1 = evaluate_model(trained_model, val_loader, device=device)
            
            tuning_results.append({
                'learning_rate': lr,
                'weight_decay': wd,
                'epochs': epochs,
                'betas': betas,
                'ol_activation': ol_activation,
                'cost_function': criterion_name,
                'validation_accuracy': val_accuracy,
                'validation_f1_score': val_f1
            })
        except Exception as e:
            print(f"Exception during tuning with parameters LR: {lr}, WD: {wd}, Epochs: {epochs}, Betas: {betas}, OL Activation: {ol_activation}, Cost Func: {criterion_name}")
            print(f"Error: {e}")

    write_results_to_csv(tuning_results, filename=tuning_filename)
    
    if not tuning_results:
        raise ValueError("No tuning results were obtained. Please check the tuning process.")
    
    # best hyperparameters based on validation accuracy
    best_result = max(tuning_results, key=lambda x: x['validation_accuracy'])
    print(f"Best Tuning Result: {best_result}")
    
    return best_result


def train_model(model, train_loader, val_loader, criterion_name='cross_entropy', optimizer_name='adam', learning_rate=0.001, weight_decay=0, betas=(0.9, 0.999), epochs=50, patience=10, device='cuda'):
    model.to(device)
    
    if criterion_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == 'nll_loss':
        criterion = nn.NLLLoss()
    else:
        raise ValueError(f"Unsupported cost function: {criterion_name}")
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict()
    
    for epoch in range(epochs):
        # training phase
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
    
        val_loss /= len(val_loader.dataset)
    
        # improvement?
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # load best model state
    model.load_state_dict(best_model_state)
    return model
