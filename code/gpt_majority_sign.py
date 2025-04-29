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
from privacy_methods import dp_sign_by_sigma, dp_by_sigma, find_min_sigma

code_version = "mlp_292"
# Global path for saving figures
FIGS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figs', code_version))
# Create directory for saving plots
os.makedirs(FIGS_PATH, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    'num_workers': 3,
    'batch_size': 128,
    'iterations_per_epoch': 100,
    'final_epsilon': 10.0,
    'final_power_in_delta': 1.1,
    'alpha_min': 2,
    'alpha_max': 20,
    'sigma_min': 0.1,
    'sigma_max': 2,
    'model_type': 'mlp',  # or 'cnn'
    'algorithms': {
        'signsgd': {
            'enabled': False,
            'num_epochs': 100,
            'lr_scheduler': lambda t: 0.01 / (t + 1)**(1/3),  # Optimal learning rate for perceptron
            'poisson_qn': 30,
            'color': 'b'
        },
        'dpsignsgd': {
            'enabled': True,
            'num_epochs': 100,
            # Learning rate scheduler with warmup: linear warmup for first 100 iterations, then decay
            'lr_scheduler': lambda t: 0.005 * min(1.0, (t+1)/500) / (t + 1)**(1/5),  # Warmup + optimal LR
            'poisson_qn': 100, # 200 and 1 --> 10 and 20
            'repeat_num': 1,
            'clipping_level': lambda t: 4, # lambda t: 10 / (t + 1)**(1/6),  # Dynamic clipping level
            'color': 'o'
        },
        'dpsgd': {
            'enabled': True,
            'num_epochs': 40,
            'lr_scheduler': lambda t: 0.01 / (t + 1)**(1/3),  # Optimal learning rate for perceptron
            'poisson_qn': 200,
            'clipping_level': lambda t: 4, # 10 / (t + 1)**(1/6),  # Dynamic clipping level
            'color': 'y'
        },
        'fedsgd': {
            'enabled': False,
            'num_epochs': 40,
            'lr_scheduler': lambda t: 0.01 / (t + 1)**(1/3),  # Constant learning rate
            'poisson_qn': 100,
            'color': 'r'
        }
    }
}

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10): # 784 = 28*28
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self._initialize_weights()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)


# Define CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        self._initialize_weights()
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

# Base Worker class
class BaseWorker:
    def __init__(self, data_loader, device, model_type='mlp', lr_scheduler=None, poisson_sampling_prob=None):
        if model_type == 'mlp':
            self.model = MLP().to(device)
        elif model_type == 'cnn':
            self.model = CNN().to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.data_loader = data_loader
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr_scheduler = lr_scheduler
        self.iter_count = 0
        self.batch_size = data_loader.batch_size
        self.dataset_size = len(data_loader.dataset)
        self.poisson_sampling_prob = poisson_sampling_prob

    def get_learning_rate(self):
        if self.lr_scheduler is None:
            return 0.05
        return self.lr_scheduler(self.iter_count)

    def compute_gradients(self):
        self.model.zero_grad()
        
        # Get batch
        images, labels = next(iter(self.data_loader))
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Apply Poisson sampling if enabled
        if self.poisson_sampling_prob is not None:
            mask = torch.rand(images.size(0), device=self.device) < self.poisson_sampling_prob
            images = images[mask]
            labels = labels[mask]
            if len(images) == 0:
                return [torch.zeros_like(param.data) for param in self.model.parameters()]
        
        # Forward pass
        outputs = self.model(images)
        loss = self.loss_fn(outputs, labels)
        
        
        # Backward pass
        loss.backward()
        
        # Get gradients
        gradients = []
        real_grads_num = 0
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
                real_grads_num += 1
            else:
                gradients.append(torch.zeros_like(param.data))
        # print(real_grads_num)
        return gradients

# SignSGD Worker
class SignSGDWorker(BaseWorker):
    def __init__(self, data_loader, device, model_type='mlp', lr_scheduler=None, poisson_sampling_prob=None):
        super().__init__(data_loader, device, model_type, lr_scheduler, poisson_sampling_prob)

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

# DPSGD Worker
class DPSGDWorker(BaseWorker):
    def __init__(self, data_loader, device, sigma, clipping_level_fn, model_type='mlp', lr_scheduler=None, poisson_sampling_prob=None):
        super().__init__(data_loader, device, model_type, lr_scheduler, poisson_sampling_prob)
        self.sigma = sigma
        self.clipping_level_fn = clipping_level_fn
        
    def compute_noise_grads(self):
        gradients = self.compute_gradients() # Bernoulli sampling is included
        dp_grads = []
        current_clipping_level = self.clipping_level_fn(self.iter_count)
        
        # Calculate and store gradient norms
        for grad in gradients:
            grad_np = grad.cpu().numpy()
            dp_grad = dp_by_sigma(grad_np, current_clipping_level, self.sigma)
            dp_grads.append(torch.from_numpy(dp_grad).to(self.device))
        
        return dp_grads

    def apply_majority_update(self, avg_grads):
        with torch.no_grad():
            if avg_grads is not None:
                lr = self.get_learning_rate()
                for param, grad in zip(self.model.parameters(), avg_grads):
                    param.add_(-lr * grad)
                self.iter_count += 1

# DPSignSGD Worker
class DPSignSGDWorker(BaseWorker):
    def __init__(self, data_loader, device, sigma, clipping_level_fn, repeat_num=1,
                 model_type='mlp', lr_scheduler=None, poisson_sampling_prob=None):
        super().__init__(data_loader, device, model_type, lr_scheduler, poisson_sampling_prob)
        self.sigma = sigma
        self.clipping_level_fn = clipping_level_fn
        self.repeat_num = repeat_num

    def compute_grad_signs(self):
        dp_grads = []
        current_clipping_level = self.clipping_level_fn(self.iter_count)
        
        # Perform multiple repetitions with different samplings
        dp_grad_sum = None
        
        for _ in range(self.repeat_num):
            # Get fresh gradients with new Poisson sampling
            gradients = self.compute_gradients()  # This includes new Poisson sampling
            
            # Calculate and store gradient norms
            for num, grad in enumerate(gradients):
                grad_np = grad.cpu().numpy()
                
                # Apply DP with new noise
                dp_grad = dp_by_sigma(grad_np, current_clipping_level, self.sigma)
                
                # Initialize or add to sum
                if dp_grad_sum is None:
                    dp_grad_sum = [None] * len(gradients)
                if dp_grad_sum[num] is None:
                    dp_grad_sum[num] = dp_grad
                else:
                    dp_grad_sum[num] += dp_grad
        
        # Take average and convert to sign
        for grad_sum in dp_grad_sum:
            dp_grad_avg = grad_sum / self.repeat_num
            dp_grad_sign = np.sign(dp_grad_avg)
            dp_grads.append(torch.from_numpy(dp_grad_sign).to(self.device))
        
        return dp_grads

    def apply_majority_update(self, majority_signs):
        with torch.no_grad():
            if majority_signs is not None:
                lr = self.get_learning_rate()
                for param, sign in zip(self.model.parameters(), majority_signs):
                    param.add_(-lr * sign)
                self.iter_count += 1

# FedSGD Worker
class FedSGDWorker(BaseWorker):
    def __init__(self, data_loader, device, model_type='mlp', lr_scheduler=None, poisson_sampling_prob=None):
        super().__init__(data_loader, device, model_type, lr_scheduler, poisson_sampling_prob)

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
    print(f"Worker sizes: {worker_sizes}")
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
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for iter in range(iterations_per_epoch):
            if optimizer_type == 'signsgd' or optimizer_type == 'dpsignsgd':
                all_signs = []
                for worker in workers:
                    signs = worker.compute_grad_signs()
                    all_signs.append(signs)
                majority_signs = majority_vote(all_signs)
                for worker in workers:
                    worker.apply_majority_update(majority_signs)
            elif optimizer_type == 'dpsgd':
                all_gradients = []
                for worker in workers:
                    gradients = worker.compute_noise_grads()
                    all_gradients.append(gradients)
                
                avg_gradients = []
                for i in range(len(all_gradients[0])):
                    stacked = torch.stack([grads[i] for grads in all_gradients])
                    avg_gradients.append(stacked.mean(dim=0))
                
                for worker in workers:
                    worker.apply_majority_update(avg_gradients)
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
    train_dataset = torchvision.datasets.MNIST(root='../data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='../data',
                                            train=False,
                                            transform=transforms.ToTensor())
    
    # Create workers
    worker_datasets = create_workers_data(train_dataset, config['num_workers'])
    worker_loaders = [DataLoader(dataset, batch_size=config['batch_size'], shuffle=True) 
                     for dataset in worker_datasets]
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Run algorithms
    results = {}
    
    if config['algorithms']['signsgd']['enabled']:
        print("\nRunning SignSGD with Majority Vote...")
        workers = [SignSGDWorker(loader, device, 
                               model_type=config['model_type'],
                               lr_scheduler=config['algorithms']['signsgd']['lr_scheduler'],
                               poisson_sampling_prob=config['algorithms']['signsgd']['poisson_qn'] / len(loader.dataset)) 
                  for loader in worker_loaders]
        results['signsgd'] = run_training('signsgd', workers, test_loader, device, 
                                        config['algorithms']['signsgd']['num_epochs'], config['iterations_per_epoch'])
    
    if config['algorithms']['dpsignsgd']['enabled']:
        print("\nRunning DPSignSGD with Majority Vote...")
        # Calculate q and sigma for each worker
        dpsignsgd_workers = []
        for loader in worker_loaders:
            q = config['algorithms']['dpsignsgd']['poisson_qn'] / len(loader.dataset)
            total_iterations = config['algorithms']['dpsignsgd']['num_epochs'] * config['iterations_per_epoch']
            sigma = find_min_sigma(q, total_iterations * config['algorithms']['dpsignsgd']['repeat_num']* 4,  # 4 parameters of MLP and 8 parameters of CNN 
                                 config['final_epsilon'], 1/(len(loader.dataset))**config['final_power_in_delta'],
                                 config['alpha_min'], config['alpha_max'],
                                 config['sigma_min'], config['sigma_max'])
            print(f"Sigma is {sigma}")
            
            worker = DPSignSGDWorker(loader, device,
                                   sigma=sigma,
                                   clipping_level_fn=config['algorithms']['dpsignsgd']['clipping_level'],
                                   model_type=config['model_type'],
                                   lr_scheduler=config['algorithms']['dpsignsgd']['lr_scheduler'],
                                   poisson_sampling_prob=q,
                                   repeat_num=config['algorithms']['dpsignsgd']['repeat_num'])
            dpsignsgd_workers.append(worker)
        
        results['dpsignsgd'] = run_training('dpsignsgd', dpsignsgd_workers, test_loader, device,
                                          config['algorithms']['dpsignsgd']['num_epochs'], config['iterations_per_epoch'])
        
        '''
        # Plot gradient norm percentiles over time
        plt.figure(figsize=(12, 6))
        for i, worker in enumerate(dpsignsgd_workers):
            if worker.norm_percentiles:
                plt.plot(range(len(worker.norm_percentiles)), worker.norm_percentiles,
                        label=f'Worker {i+1} 90th Percentile')
        
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Norm (90th Percentile)')
        plt.title('Sliding Window (50 last gradients) 90th Percentile of Gradient Norms')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(FIGS_PATH, 'gradient_norm_percentiles.png'))
        plt.close()
        
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
                plt.savefig(os.path.join(FIGS_PATH, f'dpsignsgd_worker_{i+1}_grad_norms.png'))
                plt.close()
        '''
    
    if config['algorithms']['dpsgd']['enabled']:
        print("\nRunning DPSGD...")
        dp_workers = []
        for loader in worker_loaders:
            q = config['algorithms']['dpsgd']['poisson_qn'] / len(loader.dataset)
            total_iterations = config['algorithms']['dpsgd']['num_epochs'] * config['iterations_per_epoch']
            sigma = find_min_sigma(q, total_iterations * 4,  # 4 parameters of MLP and 8 parameters of CNN 
                                 config['final_epsilon'], 1/(len(loader.dataset))**config['final_power_in_delta'],
                                 config['alpha_min'], config['alpha_max'],
                                 config['sigma_min'], config['sigma_max'])
            print(f"Sigma is {sigma}")
            
            worker = DPSGDWorker(loader, device,
                                   sigma=sigma,
                                   clipping_level_fn=config['algorithms']['dpsgd']['clipping_level'],
                                   model_type=config['model_type'],
                                   lr_scheduler=config['algorithms']['dpsgd']['lr_scheduler'],
                                   poisson_sampling_prob=q)
            dp_workers.append(worker)
            
        results['dpsgd'] = run_training('dpsgd', dp_workers, test_loader, device,
                                       config['algorithms']['dpsgd']['num_epochs'], config['iterations_per_epoch'])
    
    if config['algorithms']['fedsgd']['enabled']:
        print("\nRunning FedSGD...")
        workers = [FedSGDWorker(loader, device,
                              model_type=config['model_type'],
                              lr_scheduler=config['algorithms']['fedsgd']['lr_scheduler'],
                              poisson_sampling_prob=config['algorithms']['fedsgd']['poisson_qn'] / len(loader.dataset))
                  for loader in worker_loaders]
        results['fedsgd'] = run_training('fedsgd', workers, test_loader, device,
                                       config['algorithms']['fedsgd']['num_epochs'], config['iterations_per_epoch'])
    
    # Create plots
    max_epochs = max([config['algorithms'][algo]['num_epochs'] 
                     for algo in results if config['algorithms'][algo]['enabled']])
    epochs = range(max_epochs + 1)
    
    # Plot metrics
    metrics = ['test_accuracies', 'test_losses']
    time_metrics = ['test_accuracies', 'test_losses']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for algo in results:
            if config['algorithms'][algo]['enabled']:
                if metric == 'test_losses':
                    plt.plot(range(len(results[algo][metric])), results[algo][metric], 
                            f"{config['algorithms'][algo]['color']}-",
                            label=f'{algo.upper()} Test Loss')
                    
                else:
                    plt.plot(range(len(results[algo][metric])), results[algo][metric], 
                            f"{config['algorithms'][algo]['color']}-",
                            label=f'{algo.upper()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Federated Learning: {metric.replace("_", " ").title()} vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(FIGS_PATH, f'{metric}_vs_epochs_comparison.png'))
        plt.close()
    
    for metric in time_metrics:
        plt.figure(figsize=(12, 6))
        for algo in results:
            if config['algorithms'][algo]['enabled']:
                if metric == 'test_losses':
                    plt.plot(results[algo]['times'], results[algo][metric],
                            f"{config['algorithms'][algo]['color']}-",
                            label=f'{algo.upper()} Test Loss')
                    
                else:
                    plt.plot(results[algo]['times'], results[algo][metric],
                            f"{config['algorithms'][algo]['color']}-",
                            label=f'{algo.upper()}')
        plt.xlabel('Time (seconds)')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Federated Learning: {metric.replace("_", " ").title()} vs Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(FIGS_PATH, f'{metric}_vs_time_comparison.png'))
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
    plt.savefig(os.path.join(FIGS_PATH, 'learning_rate_schedule.png'))
    plt.close()
    
    # Print final results
    print("\nFinal Results:")
    for algo in results:
        if config['algorithms'][algo]['enabled']:
            print(f"{algo.upper()}:")
            print(f"  Final Test Accuracy: {results[algo]['test_accuracies'][-1]:.2f}%")
    
    print(f"\nPlots saved in '{FIGS_PATH}' directory")

if __name__ == "__main__":
    # Example configuration
    config = DEFAULT_CONFIG.copy()
    
    # Run experiment with configuration
    run_experiment(config)
