from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "9.0.0")
device_name = "cuda:0"
#device = "cpu"
import torch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sign_optimizer import SignSGD
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from torch.utils.data import random_split, DataLoader
from datetime import datetime

# Create directory for saving plots
os.makedirs('figs/mlp_23', exist_ok=True)

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

def create_workers_data(train_dataset, num_workers=10):
    """Split the training data among workers"""
    total_size = len(train_dataset)
    worker_sizes = [total_size // num_workers] * num_workers
    worker_sizes[-1] += total_size % num_workers  # Add remainder to last worker
    worker_datasets = random_split(train_dataset, worker_sizes)
    return worker_datasets

def train_local_model(model, train_loader, num_epochs, device):
    """Train local model and return gradients"""
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Compute gradients
            loss.backward()
            
            # Get gradients
            gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.data.clone())
                else:
                    gradients.append(torch.zeros_like(param.data))
            
            # Clear gradients
            model.zero_grad()
            
            return gradients  # Return gradients after first batch

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

def federated_training(num_workers=10, num_rounds=5, local_epochs=1):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor())
    
    # Create worker datasets
    worker_datasets = create_workers_data(train_dataset, num_workers)
    worker_loaders = [DataLoader(dataset, batch_size=64, shuffle=True) for dataset in worker_datasets]
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize global model and optimizer
    global_model = MLP().to(device)
    optimizer = SignSGD(global_model.parameters(), lr=0.00001, num_workers=num_workers)
    
    # Lists to store metrics
    test_losses = []
    test_accuracies = []
    times = []
    start_time = time.time()
    
    # Federated training
    for round in range(num_rounds):
        print(f"\nRound {round+1}/{num_rounds}")
        
        # Local training and gradient collection
        for worker_id in range(num_workers):
            print(f"Worker {worker_id+1} computing gradients...")
            local_model = MLP().to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            # Get gradients from local model
            gradients = train_local_model(local_model, worker_loaders[worker_id], local_epochs, device)
            
            # Add gradients to optimizer for majority vote
            for grad in gradients:
                optimizer.add_gradient(grad)
        
        # Update global model using majority vote
        optimizer.step()
        
        # Evaluate global model
        test_loss, test_accuracy = evaluate_model(global_model, test_loader, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        times.append(time.time() - start_time)
        
        print(f"Round {round+1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    return {
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'times': times
    }

# Run federated training with SignSGD and majority vote
print("\nFederated Training with SignSGD and Majority Vote...")
metrics = federated_training()

# Create and save plots
rounds = range(1, len(metrics['test_accuracies']) + 1)

# Accuracy vs Rounds
plt.figure(figsize=(12, 6))
plt.plot(rounds, metrics['test_accuracies'], 'b-', label='SignSGD with Majority Vote')
plt.xlabel('Communication Rounds')
plt.ylabel('Test Accuracy (%)')
plt.title('Federated Learning: Accuracy vs Communication Rounds')
plt.legend()
plt.grid(True)
plt.savefig('figs/mlp_23/federated_accuracy_vs_rounds.png')
plt.close()

# Loss vs Rounds
plt.figure(figsize=(12, 6))
plt.plot(rounds, metrics['test_losses'], 'b-', label='SignSGD with Majority Vote')
plt.xlabel('Communication Rounds')
plt.ylabel('Test Loss')
plt.title('Federated Learning: Loss vs Communication Rounds')
plt.legend()
plt.grid(True)
plt.savefig('figs/mlp_23/federated_loss_vs_rounds.png')
plt.close()

# Accuracy vs Time
plt.figure(figsize=(12, 6))
plt.plot(metrics['times'], metrics['test_accuracies'], 'b-', label='SignSGD with Majority Vote')
plt.xlabel('Time (seconds)')
plt.ylabel('Test Accuracy (%)')
plt.title('Federated Learning: Accuracy vs Time')
plt.legend()
plt.grid(True)
plt.savefig('figs/mlp_23/federated_accuracy_vs_time.png')
plt.close()

# Loss vs Time
plt.figure(figsize=(12, 6))
plt.plot(metrics['times'], metrics['test_losses'], 'b-', label='SignSGD with Majority Vote')
plt.xlabel('Time (seconds)')
plt.ylabel('Test Loss')
plt.title('Federated Learning: Loss vs Time')
plt.legend()
plt.grid(True)
plt.savefig('figs/mlp_23/federated_loss_vs_time.png')
plt.close()

print(f"\nFinal Test Accuracy: {metrics['test_accuracies'][-1]:.2f}%")
print(f"Plots saved in 'figs/mlp_23/' directory") 