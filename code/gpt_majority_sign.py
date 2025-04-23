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
from torch.utils.data import random_split, DataLoader

# Create directory for saving plots
os.makedirs('figs/mlp_23', exist_ok=True)

# Settings
NUM_WORKERS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 64

# Define MLP model (from train_mlp.py)
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

    def get_learning_rate(self):
        if self.lr_scheduler is None:
            return 0.05  # Default constant learning rate for FedSGD
        return self.lr_scheduler(self.iter_count)

    def compute_gradients(self):
        self.model.zero_grad()
        # Get first batch
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
        if lr_scheduler is None:
            # Default scheduler for SignSGD: 0.1/sqrt(T)
            lr_scheduler = lambda t: 0.1 / math.sqrt(t + 1)  # +1 to avoid division by zero
        super().__init__(data_loader, device, lr_scheduler)

    def compute_grad_signs(self):
        gradients = self.compute_gradients()
        return [torch.sign(grad) for grad in gradients]

    def apply_majority_update(self, majority_signs):
        with torch.no_grad():
            lr = self.get_learning_rate()
            for param, sign in zip(self.model.parameters(), majority_signs):
                param.add_(-lr * sign)
            self.iter_count += 1

# FedSGD Worker
class FedSGDWorker(BaseWorker):
    def __init__(self, data_loader, device, lr_scheduler=None):
        if lr_scheduler is None:
            # Default constant learning rate for FedSGD
            lr_scheduler = lambda t: 0.05
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
    """Split the training data among workers"""
    total_size = len(train_dataset)
    worker_sizes = [total_size // num_workers] * num_workers
    worker_sizes[-1] += total_size % num_workers  # Add remainder to last worker
    worker_datasets = random_split(train_dataset, worker_sizes)
    return worker_datasets

# Majority vote across workers
def majority_vote(signs_list):
    num_params = len(signs_list[0])
    result = []
    for i in range(num_params):
        stacked = torch.stack([worker_signs[i] for worker_signs in signs_list])
        vote = torch.sign(stacked.sum(dim=0))
        result.append(vote)
    return result

def run_training(optimizer_type, workers, test_loader, device, num_epochs):
    # Initialize global model for evaluation
    global_model = MLP().to(device)
    
    # Lists to store metrics
    test_losses = []
    test_accuracies = []
    times = []
    learning_rates = []  # Track learning rates
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for iter in range(300):
            if optimizer_type == 'signsgd':
                # SignSGD with majority vote
                all_signs = []
                for worker in workers:
                    signs = worker.compute_grad_signs()
                    all_signs.append(signs)
                majority_signs = majority_vote(all_signs)
                for worker in workers:
                    worker.apply_majority_update(majority_signs)
            else:  # fedsgd
                # FedSGD
                all_gradients = []
                for worker in workers:
                    gradients = worker.compute_gradients()
                    all_gradients.append(gradients)
                
                # Average gradients
                avg_gradients = []
                for i in range(len(all_gradients[0])):
                    stacked = torch.stack([grads[i] for grads in all_gradients])
                    avg_gradients.append(stacked.mean(dim=0))
                
                # Apply updates
                for worker in workers:
                    worker.apply_fed_update(avg_gradients)
            
            # Track learning rate of first worker
            learning_rates.append(workers[0].get_learning_rate())
        
        # Update global model (use first worker's model for evaluation)
        global_model.load_state_dict(workers[0].model.state_dict())
        
        # Evaluate
        test_loss, test_accuracy = evaluate_model(global_model, test_loader, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        times.append(time.time() - start_time)
        
        print(f"Epoch {epoch + 1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print(f"Current learning rate: {workers[0].get_learning_rate():.6f}")
    
    return {
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'times': times,
        'learning_rates': learning_rates
    }

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor())
    
    # Create worker datasets and data loaders
    worker_datasets = create_workers_data(train_dataset, NUM_WORKERS)
    worker_loaders = [DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) for dataset in worker_datasets]
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Define learning rate schedulers
    signsgd_scheduler = lambda t: 0.01 / math.sqrt(t + 1)  # SignSGD scheduler
    fedsgd_scheduler = lambda t: 0.05  # FedSGD scheduler
    
    # Run SignSGD with majority vote
    print("\nRunning SignSGD with Majority Vote (adaptive lr = 0.01/sqrt(T))...")
    signsgd_workers = [SignSGDWorker(loader, device, signsgd_scheduler) for loader in worker_loaders]
    signsgd_metrics = run_training('signsgd', signsgd_workers, test_loader, device, NUM_EPOCHS)
    
    # Run FedSGD
    print("\nRunning FedSGD (constant lr = 0.05)...")
    fedsgd_workers = [FedSGDWorker(loader, device, fedsgd_scheduler) for loader in worker_loaders]
    fedsgd_metrics = run_training('fedsgd', fedsgd_workers, test_loader, device, NUM_EPOCHS)
    
    # Create and save plots
    epochs = range(1, NUM_EPOCHS + 1)
    
    # Accuracy vs Epochs
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, signsgd_metrics['test_accuracies'], 'b-', label='SignSGD with Majority Vote')
    plt.plot(epochs, fedsgd_metrics['test_accuracies'], 'r-', label='FedSGD')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Federated Learning: Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/mlp_23/accuracy_vs_epochs_comparison.png')
    plt.close()
    
    # Loss vs Epochs
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, signsgd_metrics['test_losses'], 'b-', label='SignSGD with Majority Vote')
    plt.plot(epochs, fedsgd_metrics['test_losses'], 'r-', label='FedSGD')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Federated Learning: Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/mlp_23/loss_vs_epochs_comparison.png')
    plt.close()
    
    # Accuracy vs Time
    plt.figure(figsize=(12, 6))
    plt.plot(signsgd_metrics['times'], signsgd_metrics['test_accuracies'], 'b-', label='SignSGD with Majority Vote')
    plt.plot(fedsgd_metrics['times'], fedsgd_metrics['test_accuracies'], 'r-', label='FedSGD')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Accuracy (%)')
    plt.title('Federated Learning: Accuracy vs Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/mlp_23/accuracy_vs_time_comparison.png')
    plt.close()
    
    # Loss vs Time
    plt.figure(figsize=(12, 6))
    plt.plot(signsgd_metrics['times'], signsgd_metrics['test_losses'], 'b-', label='SignSGD with Majority Vote')
    plt.plot(fedsgd_metrics['times'], fedsgd_metrics['test_losses'], 'r-', label='FedSGD')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Loss')
    plt.title('Federated Learning: Loss vs Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/mlp_23/loss_vs_time_comparison.png')
    plt.close()
    
    # Learning Rate vs Iterations
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(signsgd_metrics['learning_rates'])), signsgd_metrics['learning_rates'], 'b-', label='SignSGD with Majority Vote')
    plt.axhline(y=0.05, color='r', linestyle='-', label='FedSGD (constant)')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.savefig('figs/mlp_23/learning_rate_schedule.png')
    plt.close()
    
    print(f"\nFinal Test Accuracy:")
    print(f"SignSGD with Majority Vote: {signsgd_metrics['test_accuracies'][-1]:.2f}%")
    print(f"FedSGD: {fedsgd_metrics['test_accuracies'][-1]:.2f}%")
    print(f"Plots saved in 'figs/mlp_23/' directory")

if __name__ == "__main__":
    main()
