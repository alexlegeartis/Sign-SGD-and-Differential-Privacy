import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import create_model
from privacy_methods import dp_by_sigma

class BaseWorker:
    def __init__(self, data_loader, device, model_type='mlp', lr_scheduler=None, poisson_sampling_prob=None):
        self.model = create_model(model_type, device)
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
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
            else:
                gradients.append(torch.zeros_like(param.data))
        
        # Print number of parameters being sent
        # print(f"Number of gradients being sent: {len(gradients)}")
        return gradients

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
                    param.add_(-lr * sign.to(self.device))
                self.iter_count += 1

class DPSGDWorker(BaseWorker):
    def __init__(self, data_loader, device, sigma, clipping_level_fn, model_type='mlp', lr_scheduler=None, poisson_sampling_prob=None):
        super().__init__(data_loader, device, model_type, lr_scheduler, poisson_sampling_prob)
        self.sigma = sigma
        self.clipping_level_fn = clipping_level_fn
        
    def compute_noise_grads(self):
        gradients = self.compute_gradients()
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
                    param.add_(-lr * grad.to(self.device))
                self.iter_count += 1

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
            gradients = self.compute_gradients()
            
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
                    param.add_(-lr * sign.to(self.device))
                self.iter_count += 1

class FedSGDWorker(BaseWorker):
    def __init__(self, data_loader, device, model_type='mlp', lr_scheduler=None, poisson_sampling_prob=None):
        super().__init__(data_loader, device, model_type, lr_scheduler, poisson_sampling_prob)

    def apply_fed_update(self, global_gradients):
        with torch.no_grad():
            lr = self.get_learning_rate()
            for param, grad in zip(self.model.parameters(), global_gradients):
                param.add_(-lr * grad)
            self.iter_count += 1 