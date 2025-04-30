#!/usr/bin/python3
#from os import putenv
# putenv("HSA_OVERRIDE_GFX_VERSION", "9.0.0")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data import random_split, DataLoader
import logging
from datetime import datetime
import json
import shutil
import sys
import torch.multiprocessing as mp

from models import create_model
from privacy_methods import find_min_sigma
from workers import SignSGDWorker, DPSGDWorker, DPSignSGDWorker, FedSGDWorker

# Global paths
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGS_PATH = os.path.join(BASE_PATH, 'figs')
LOG_PATH = os.path.join(BASE_PATH, 'logs')

# Learning rate scheduler functions
def lr_scheduler_1(t):
    return 0.01 / (t + 1)**(1/3)

def lr_scheduler_2(t):
    return min(1, (t+1)/500) * 0.0005 / (t+1)**(0.08)

def lr_scheduler_3(t):
    return min(1, (t+1)/1000) * 0.0005

def lr_scheduler_many_workers(t):
    return 0.0005

def clipping_level_4(t):
    return 4

def setup_logging(code_version):
    # Create base directories
    os.makedirs(FIGS_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    
    # Create timestamp in readable format (YYYY-MM-DD_HH-MM-SS)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create experiment-specific directories
    experiment_path = os.path.join(LOG_PATH, code_version + '_' + timestamp)
    os.makedirs(experiment_path, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(experiment_path, f'experiment{timestamp}.log')
    print_handler = logging.StreamHandler(sys.stdout)  # Send logs to stdout like print()
    print_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            print_handler
        ]
    )
    
    logging.info(f"Created experiment directory: {experiment_path}")
    logging.info(f"Logs will be saved to: {log_file}")
    
    return FIGS_PATH, experiment_path, timestamp

def backup_config(config_path, experiment_path, timestamp):
    # Create backup of config file in the experiment directory
    config_backup_path = os.path.join(experiment_path, f'config_{timestamp}.json')
    shutil.copy2(config_path, config_backup_path)
    logging.info(f"Config file backed up to: {config_backup_path}")

def plot_results(results, config, figs_path):
    # Create plots
    max_epochs = max([algo['num_epochs'] for algo in config['algorithms'] if algo['enabled']])
    epochs = range(max_epochs + 1)
    
    # Plot metrics
    metrics = ['test_accuracies', 'test_losses']
    time_metrics = ['test_accuracies', 'test_losses']
    
    # Get a list of colors from matplotlib's default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for i, algo in enumerate([a for a in config['algorithms'] if a['enabled']]):
            if metric == 'test_losses':
                plt.plot(range(len(results[algo['label']][metric])), results[algo['label']][metric], 
                        color=colors[i % len(colors)],
                        label=f"{algo['label']} Test Loss")
            else:
                plt.plot(range(len(results[algo['label']][metric])), results[algo['label']][metric], 
                        color=colors[i % len(colors)],
                        label=algo['label'])
        plt.xlabel('Epochs')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Federated Learning: {metric.replace("_", " ").title()} vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figs_path, f'{metric}_vs_epochs_comparison.png'))
        plt.close()
    
    for metric in time_metrics:
        plt.figure(figsize=(12, 6))
        for i, algo in enumerate([a for a in config['algorithms'] if a['enabled']]):
            if metric == 'test_losses':
                plt.plot(results[algo['label']]['times'], results[algo['label']][metric],
                        color=colors[i % len(colors)],
                        label=f"{algo['label']} Test Loss")
            else:
                plt.plot(results[algo['label']]['times'], results[algo['label']][metric],
                        color=colors[i % len(colors)],
                        label=algo['label'])
        plt.xlabel('Time (seconds)')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Federated Learning: {metric.replace("_", " ").title()} vs Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figs_path, f'{metric}_vs_time_comparison.png'))
        plt.close()
    
    # Learning Rate Schedule
    plt.figure(figsize=(12, 6))
    for i, algo in enumerate([a for a in config['algorithms'] if a['enabled']]):
        plt.plot(range(len(results[algo['label']]['learning_rates'])), 
                results[algo['label']]['learning_rates'],
                color=colors[i % len(colors)],
                label=algo['label'])
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figs_path, 'learning_rate_schedule.png'))
    plt.close()
    
    logging.info(f"Plots saved in '{figs_path}' directory")

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return test_loss / len(test_loader), 100 * correct / total

def create_workers_data(train_dataset, num_workers=10):
    total_size = len(train_dataset)
    worker_sizes = [total_size // num_workers] * num_workers
    worker_sizes[-1] += total_size % num_workers
    print(f"Worker sizes: {worker_sizes}")
    worker_datasets = random_split(train_dataset, worker_sizes)
    return worker_datasets

def majority_vote(signs_list):
    num_params = len(signs_list[0])
    result = []
    for i in range(num_params):
        stacked = torch.stack([worker_signs[i] for worker_signs in signs_list])
        vote = torch.sign(stacked.sum(dim=0)).cpu()
        result.append(vote)
    return result

def compute_worker_grads_mp(worker, grad_type):
    if grad_type == 'signsgd' or grad_type == 'dpsignsgd':
        signs = worker.compute_grad_signs()
        return [sign.cpu() for sign in signs]  # Move to CPU before returning
    elif grad_type == 'dpsgd':
        grads = worker.compute_noise_grads()
        return [grad.cpu() for grad in grads]  # Move to CPU before returning
    else:  # fedsgd
        grads = worker.compute_gradients()
        return [grad.cpu() for grad in grads]  # Move to CPU before returning

def apply_worker_update_mp(worker, update_data, update_type):
    if update_type == 'signsgd' or update_type == 'dpsignsgd':
        worker.apply_majority_update(update_data)
    elif update_type == 'dpsgd':
        worker.apply_majority_update(update_data)
    else:  # fedsgd
        worker.apply_fed_update(update_data)

def run_training(optimizer_type, workers, test_loader, device, num_epochs, iterations_per_epoch):
    global_model = workers[0].model.to(device)
    
    metrics = {
        'test_losses': [],
        'test_accuracies': [],
        'times': [],
        'learning_rates': []
    }
    
    start_time = time.time()
    
    # Initial evaluation
    test_loss, test_accuracy = evaluate_model(global_model, test_loader, device)
    metrics['test_losses'].append(test_loss)
    metrics['test_accuracies'].append(test_accuracy)
    metrics['times'].append(0.0)
    
    # Calculate how many workers per process
    num_workers = len(workers)
    num_processes = 10
    
    # Create worker groups
    worker_groups = [workers[i:i + num_processes] 
                    for i in range(0, num_workers, num_processes)]
    print(f"Groups: {len(worker_groups)}")
    
    with mp.Pool(processes=num_processes) as pool:
        for epoch in range(num_epochs):
            logging.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            for iter in range(iterations_per_epoch):
                # Compute gradients in parallel
                all_grads = []
                for group in worker_groups:
                    group_grads = pool.starmap(compute_worker_grads_mp,
                                             [(worker, optimizer_type) for worker in group])
                    all_grads.extend(group_grads)
                
                if optimizer_type == 'signsgd' or optimizer_type == 'dpsignsgd':
                    # Compute majority vote
                    majority_signs = majority_vote(all_grads)
                    # Apply updates directly
                    for worker in workers:
                        apply_worker_update_mp(worker, majority_signs, optimizer_type)
                elif optimizer_type == 'dpsgd':
                    # Compute average gradients
                    avg_gradients = []
                    for i in range(len(all_grads[0])):
                        stacked = torch.stack([grads[i] for grads in all_grads])
                        avg_gradients.append(stacked.mean(dim=0))
                    # Apply updates directly
                    for worker in workers:
                        apply_worker_update_mp(worker, avg_gradients, optimizer_type)
                else:  # fedsgd
                    # Compute average gradients
                    avg_gradients = []
                    for i in range(len(all_grads[0])):
                        stacked = torch.stack([grads[i] for grads in all_grads])
                        avg_gradients.append(stacked.mean(dim=0))
                    # Apply updates directly
                    for worker in workers:
                        apply_worker_update_mp(worker, avg_gradients, optimizer_type)
                
                metrics['learning_rates'].append(workers[0].get_learning_rate())
            
            # Update global model
            global_model.load_state_dict(workers[0].model.state_dict())
            
            test_loss, test_accuracy = evaluate_model(global_model, test_loader, device)
            metrics['test_losses'].append(test_loss)
            metrics['test_accuracies'].append(test_accuracy)
            metrics['times'].append(time.time() - start_time)
            
            logging.info(f"Test Loss: {test_loss:.4f}")
            logging.info(f"Test Accuracy: {test_accuracy:.2f}%")
            logging.info(f"Learning rate: {workers[0].get_learning_rate():.6f}")
    
    return metrics

# Load configuration from file
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Convert string lambda expressions to actual functions
    for algo in config['algorithms']:
        if 'lr_scheduler' in algo:
            algo['lr_scheduler'] = eval(algo['lr_scheduler'])
        if 'clipping_level' in algo:
            algo['clipping_level'] = eval(algo['clipping_level'])
    
    return config

def run_experiment(config):
    # Setup logging and get paths
    figs_path, experiment_path, timestamp = setup_logging(config['code_version'])
    
    # Backup config
    config_path = os.path.join(os.path.dirname(__file__), 'config_cnn.json')
    backup_config(config_path, experiment_path, timestamp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set up multiprocessing
    # mp.set_start_method('spawn')
    mp.set_start_method('fork', force=True)  # 'fork' avoids pickling issues (on Unix)
    logging.info(f"Using device: {device}")

    # Load dataset based on config
    dataset_name = config.get('dataset', 'mnist')
    logging.info(f"Loading {dataset_name} dataset")
    
    if dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root='../data',
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
        test_dataset = torchvision.datasets.MNIST(root='../data',
                                                train=False,
                                                transform=transforms.ToTensor())
    else:  # cifar10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='../data',
                                                   train=True,
                                                   transform=transform,
                                                   download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='../data',
                                                  train=False,
                                                  transform=transform)
    
    # Create workers
    worker_datasets = create_workers_data(train_dataset, config['num_workers'])
    worker_loaders = [DataLoader(dataset, batch_size=config['batch_size'], shuffle=True) 
                     for dataset in worker_datasets]
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Calculate grads_per_iter by creating a temporary model
    temp_model = create_model(config['model_type'], device, dataset_name)
    grads_per_iter = sum(1 for _ in temp_model.parameters())
    logging.info(f"Number of gradients per iteration: {grads_per_iter}")
    
    # Delete temporary model
    del temp_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Run algorithms
    results = {}
    
    for algo in config['algorithms']:
        if not algo['enabled']:
            continue
            
        logging.info(f"\nRunning {algo['label']}...")
        
        if algo['type'] == 'signsgd':
            workers = [SignSGDWorker(loader, device, 
                                   model_type=config['model_type'],
                                   lr_scheduler=algo['lr_scheduler'],
                                   poisson_sampling_prob=algo['poisson_qn'] / len(loader.dataset)) 
                      for loader in worker_loaders]
            results[algo['label']] = run_training('signsgd', workers, test_loader, device, 
                                               algo['num_epochs'], config['iterations_per_epoch'])
        
        elif algo['type'] == 'dpsignsgd':
            dpsignsgd_workers = []
            for loader in worker_loaders:
                q = algo['poisson_qn'] / len(loader.dataset)
                total_iterations = algo['num_epochs'] * config['iterations_per_epoch']
                sigma = find_min_sigma(q, total_iterations * algo['repeat_num'] * grads_per_iter,
                                     algo['final_epsilon'], 1/(len(loader.dataset))**algo['final_power_in_delta'],
                                     config['alpha_min'], config['alpha_max'],
                                     config['sigma_min'], config['sigma_max'])
                logging.info(f"Sigma for {algo['label']}: {sigma}")
                
                worker = DPSignSGDWorker(loader, device,
                                       sigma=sigma,
                                       clipping_level_fn=algo['clipping_level'],
                                       model_type=config['model_type'],
                                       lr_scheduler=algo['lr_scheduler'],
                                       poisson_sampling_prob=q,
                                       repeat_num=algo['repeat_num'])
                dpsignsgd_workers.append(worker)
            
            results[algo['label']] = run_training('dpsignsgd', dpsignsgd_workers, test_loader, device,
                                               algo['num_epochs'], config['iterations_per_epoch'])
        
        elif algo['type'] == 'dpsgd':
            dp_workers = []
            for loader in worker_loaders:
                q = algo['poisson_qn'] / len(loader.dataset)
                total_iterations = algo['num_epochs'] * config['iterations_per_epoch']
                sigma = find_min_sigma(q, total_iterations * grads_per_iter,
                                     algo['final_epsilon'], 1/(len(loader.dataset))**algo['final_power_in_delta'],
                                     config['alpha_min'], config['alpha_max'],
                                     config['sigma_min'], config['sigma_max'])
                logging.info(f"Sigma for {algo['label']}: {sigma}")
                
                worker = DPSGDWorker(loader, device,
                                   sigma=sigma,
                                   clipping_level_fn=algo['clipping_level'],
                                   model_type=config['model_type'],
                                   lr_scheduler=algo['lr_scheduler'],
                                   poisson_sampling_prob=q)
                dp_workers.append(worker)
            
            results[algo['label']] = run_training('dpsgd', dp_workers, test_loader, device,
                                               algo['num_epochs'], config['iterations_per_epoch'])
        
        elif algo['type'] == 'fedsgd':
            workers = [FedSGDWorker(loader, device,
                                  model_type=config['model_type'],
                                  lr_scheduler=algo['lr_scheduler'],
                                  poisson_sampling_prob=algo['poisson_qn'] / len(loader.dataset))
                      for loader in worker_loaders]
            results[algo['label']] = run_training('fedsgd', workers, test_loader, device,
                                               algo['num_epochs'], config['iterations_per_epoch'])
    
    # Plot results
    plot_results(results, config, figs_path)
    
    # Print final results
    logging.info("\nFinal Results:")
    for algo in config['algorithms']:
        if not algo['enabled']:
            continue
        logging.info(f"{algo['label']}:")
        logging.info(f"  Final Test Accuracy: {results[algo['label']]['test_accuracies'][-1]:.2f}%")
    
    logging.info(f"Log file saved in '{experiment_path}'")

if __name__ == "__main__":
    # Load configuration from file
    config_path = os.path.join(os.path.dirname(__file__), 'config_cnn.json')
    config = load_config(config_path)
    
    # Run experiment with configuration
    run_experiment(config)
