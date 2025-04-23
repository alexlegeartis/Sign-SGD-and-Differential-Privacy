from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "9.0.0")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
import math
import numpy as np
from torch.utils.data import random_split, DataLoader
from privacy_methods import dp_sign_by_sigma, find_min_sigma

# Create directory for saving plots
os.makedirs('figs/mlp_23', exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    'num_workers': 10,
    'batch_size': 64,
    'num_epochs': 20,
    'iterations_per_epoch': 100,
    'final_epsilon': 1.0,
    'final_power_in_delta': 1.1,
    'alpha_min': 2,
    'alpha_max': 20,
    'sigma_min': 0.7,
    'sigma_max': 1.5,
    'algorithms': {
        'signsgd': {
            'enabled': True,
            'lr_scheduler': lambda t: 0.02 / (t + 1)**(1/4),
            'color': 'b'
        },
        'dpsignsgd': {
            'enabled': True,
            'lr_scheduler': lambda t: 0.02 / (t + 1)**(1/4),
            'color': 'g'
        },
        'fedsgd': {
            'enabled': False,
            'lr_scheduler': lambda t: 0.05,
            'color': 'r'
        }
    }
}

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Base Worker class
class BaseWorker:
    def __init__(self, data_loader, device, lr_scheduler=None):
        self.model = MLP().to(device)
        self.data_loader = data_loader
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr_scheduler = lr_scheduler
        self.iter_count = 0
        self.batch_size = data_loader.batch_size
        self.dataset_size = len(data_loader.dataset)

    def get_learning_rate(self):
        if self.lr_scheduler is None:
            return 0.05
        return self.lr_scheduler(self.iter_count)

    def compute_gradients(self):
        self.model.zero_grad()
        
        # Get batch
        images, labels = next(iter(self.data_loader))
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        loss = self.loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Get gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
            else:
                gradients.append(torch.zeros_like(param.data))
        return gradients

# SignSGD Worker
class SignSGDWorker(BaseWorker):
    def __init__(self, data_loader, device, lr_scheduler=None):
        super().__init__(data_loader, device, lr_scheduler)

    def compute_grad_signs(self):
        gradients = self.compute_gradients()
        return [torch.sign(grad) for grad in gradients]

    def apply_majority_update(self, majority_signs):
        with torch.no_grad():
            if majority_signs is not None:
                lr = self.get_learning_rate()
                for param, sign in zip(self.model.parameters(), majority_signs):
                    param.add_(-lr * sign)
                self.iter_count += 1

# DPSignSGD Worker
class DPSignSGDWorker(BaseWorker):
    def __init__(self, data_loader, device, sigma, sensitivity_measure, lr_scheduler=None):
        super().__init__(data_loader, device, lr_scheduler)
        self.sigma = sigma
        self.sensitivity_measure = sensitivity_measure
        self.gradient_norms = []

    def compute_grad_signs(self):
        gradients = self.compute_gradients()
        dp_signs = []
        for grad in gradients:
            grad_np = grad.cpu().numpy()
            grad_norm = np.linalg.norm(grad_np)
            self.gradient_norms.append(grad_norm)
            dp_sign = dp_sign_by_sigma(grad_np, self.sensitivity_measure, self.sigma)
            dp_signs.append(torch.from_numpy(dp_sign).to(self.device))
        return dp_signs

    def apply_majority_update(self, majority_signs):
        with torch.no_grad():
            if majority_signs is not None:
                lr = self.get_learning_rate()
                for param, sign in zip(self.model.parameters(), majority_signs):
                    param.add_(-lr * sign)
                self.iter_count += 1

# FedSGD Worker
class FedSGDWorker(BaseWorker):
    def __init__(self, data_loader, device, lr_scheduler=None):
        super().__init__(data_loader, device, lr_scheduler)

    def apply_fed_update(self, global_gradients):
        with torch.no_grad():
            lr = self.get_learning_rate()
            for param, grad in zip(self.model.parameters(), global_gradients):
                param.add_(-lr * grad)
            self.iter_count += 1

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
    worker_datasets = random_split(train_dataset, worker_sizes)
    return worker_datasets

def majority_vote(signs_list):
    num_params = len(signs_list[0])
    result = []
    for i in range(num_params):
        stacked = torch.stack([worker_signs[i] for worker_signs in signs_list])
        vote = torch.sign(stacked.sum(dim=0))
        result.append(vote)
    return result


def run_training(optimizer_type, workers, test_loader, device, num_epochs, iterations_per_epoch):
    global_model = MLP().to(device)
    
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
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for iter in range(iterations_per_epoch):
            if optimizer_type == 'signsgd':
                all_signs = []
                for worker in workers:
                    signs = worker.compute_grad_signs()
                    all_signs.append(signs)
                majority_signs = majority_vote(all_signs)
                for worker in workers:
                    worker.apply_majority_update(majority_signs)
            else:  # fedsgd
                all_gradients = []
                for worker in workers:
                    gradients = worker.compute_gradients()
                    all_gradients.append(gradients)
                
                avg_gradients = []
                for i in range(len(all_gradients[0])):
                    stacked = torch.stack([grads[i] for grads in all_gradients])
                    avg_gradients.append(stacked.mean(dim=0))
                
                for worker in workers:
                    worker.apply_fed_update(avg_gradients)
            
            metrics['learning_rates'].append(workers[0].get_learning_rate())
        
        global_model.load_state_dict(workers[0].model.state_dict())
        
        test_loss, test_accuracy = evaluate_model(global_model, test_loader, device)
        metrics['test_losses'].append(test_loss)
        metrics['test_accuracies'].append(test_accuracy)
        metrics['times'].append(time.time() - start_time)
        
        print(f"Epoch {epoch + 1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print(f"Current learning rate: {workers[0].get_learning_rate():.6f}")
    
    return metrics

def analyze_gradient_norms(worker):
    if not worker.gradient_norms:
        return None
    
    norms = np.array(worker.gradient_norms)
    median = np.median(norms)
    lower_percentile = np.percentile(norms, 2.5)
    upper_percentile = np.percentile(norms, 97.5)
    
    return {
        'median': median,
        'lower_95': lower_percentile,
        'upper_95': upper_percentile,
        'norms': norms
    }

def run_experiment(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor())
    
    # Create workers
    worker_datasets = create_workers_data(train_dataset, config['num_workers'])
    worker_loaders = [DataLoader(dataset, batch_size=config['batch_size'], shuffle=True) 
                     for dataset in worker_datasets]
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Calculate total iterations and q
    total_iterations = config['num_epochs'] * config['iterations_per_epoch']
    
    # Run algorithms
    results = {}
    
    if config['algorithms']['signsgd']['enabled']:
        print("\nRunning SignSGD with Majority Vote...")
        workers = [SignSGDWorker(loader, device, 
                               lr_scheduler=config['algorithms']['signsgd']['lr_scheduler']) 
                  for loader in worker_loaders]
        results['signsgd'] = run_training('signsgd', workers, test_loader, device, 
                                        config['num_epochs'], config['iterations_per_epoch'])
    
    if config['algorithms']['dpsignsgd']['enabled']:
        print("\nRunning DPSignSGD with Majority Vote...")
        # Calculate q and sigma for each worker
        dpsignsgd_workers = []
        for loader in worker_loaders:
            q = 1 / len(loader.dataset)  # Poisson probability for subsampling
            sigma = find_min_sigma(q, total_iterations, 
                                 config['final_epsilon'], 1/(len(loader.dataset))**config['final_power_in_delta'],
                                 config['alpha_min'], config['alpha_max'],
                                 config['sigma_min'], config['sigma_max'])
            
            worker = DPSignSGDWorker(loader, device,
                                   sigma=sigma,
                                   sensitivity_measure=1.0,
                                   lr_scheduler=config['algorithms']['dpsignsgd']['lr_scheduler'])
            dpsignsgd_workers.append(worker)
        
        results['dpsignsgd'] = run_training('signsgd', dpsignsgd_workers, test_loader, device,
                                          config['num_epochs'], config['iterations_per_epoch'])
        
        # Analyze gradient norms
        print("\nAnalyzing gradient norms for DPSignSGD:")
        for i, worker in enumerate(dpsignsgd_workers):
            analysis = analyze_gradient_norms(worker)
            if analysis:
                print(f"\nWorker {i + 1} gradient norm statistics:")
                print(f"Median: {analysis['median']:.4f}")
                print(f"95% interval: [{analysis['lower_95']:.4f}, {analysis['upper_95']:.4f}]")
                
                plt.figure(figsize=(10, 6))
                plt.hist(analysis['norms'], bins=50, alpha=0.7)
                plt.axvline(analysis['median'], color='r', linestyle='--', label=f'Median: {analysis["median"]:.4f}')
                plt.axvline(analysis['lower_95'], color='g', linestyle='--', label=f'2.5%: {analysis["lower_95"]:.4f}')
                plt.axvline(analysis['upper_95'], color='g', linestyle='--', label=f'97.5%: {analysis["upper_95"]:.4f}')
                plt.xlabel('Gradient Norm')
                plt.ylabel('Frequency')
                plt.title(f'DPSignSGD Worker {i + 1} Gradient Norm Distribution')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'figs/mlp_23/dpsignsgd_worker_{i+1}_grad_norms.png')
                plt.close()
    
    if config['algorithms']['fedsgd']['enabled']:
        print("\nRunning FedSGD...")
        workers = [FedSGDWorker(loader, device,
                              lr_scheduler=config['algorithms']['fedsgd']['lr_scheduler'])
                  for loader in worker_loaders]
        results['fedsgd'] = run_training('fedsgd', workers, test_loader, device,
                                       config['num_epochs'], config['iterations_per_epoch'])
    
    # Create plots
    epochs = range(config['num_epochs'] + 1)
    
    # Plot metrics
    metrics = ['test_accuracies', 'test_losses']
    time_metrics = ['test_accuracies', 'test_losses']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for algo in results:
            if config['algorithms'][algo]['enabled']:
                plt.plot(epochs, results[algo][metric], 
                        f"{config['algorithms'][algo]['color']}-",
                        label=f'{algo.upper()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Federated Learning: {metric.replace("_", " ").title()} vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figs/mlp_23/{metric}_vs_epochs_comparison.png')
        plt.close()
    
    for metric in time_metrics:
        plt.figure(figsize=(12, 6))
        for algo in results:
            if config['algorithms'][algo]['enabled']:
                plt.plot(results[algo]['times'], results[algo][metric],
                        f"{config['algorithms'][algo]['color']}-",
                        label=f'{algo.upper()}')
        plt.xlabel('Time (seconds)')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Federated Learning: {metric.replace("_", " ").title()} vs Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figs/mlp_23/{metric}_vs_time_comparison.png')
        plt.close()
    
    # Learning Rate Schedule
    plt.figure(figsize=(12, 6))
    for algo in results:
        if config['algorithms'][algo]['enabled']:
            plt.plot(range(len(results[algo]['learning_rates'])), 
                    results[algo]['learning_rates'],
                    f"{config['algorithms'][algo]['color']}-",
                    label=f'{algo.upper()}')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/mlp_23/learning_rate_schedule.png')
    plt.close()
    
    # Print final results
    print("\nFinal Results:")
    for algo in results:
        if config['algorithms'][algo]['enabled']:
            print(f"{algo.upper()}:")
            print(f"  Final Test Accuracy: {results[algo]['test_accuracies'][-1]:.2f}%")
    
    print("\nPlots saved in 'figs/mlp_23/' directory")

if __name__ == "__main__":
    # Example configuration
    config = DEFAULT_CONFIG.copy()
    
    # Run experiment with configuration
    run_experiment(config)
