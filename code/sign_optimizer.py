import torch
from torch.optim import Optimizer

class SignSGD(Optimizer):
    """
    Implements Stochastic-Sign SGD with majority vote algorithm.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        num_workers (int): number of workers for majority vote
    """
    def __init__(self, params, lr=0.01, num_workers=10):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, num_workers=num_workers)
        super(SignSGD, self).__init__(params, defaults)
        self.gradients = []
        self.current_worker = 0

    def add_gradient(self, grad):
        """Add a worker's gradient to the collection"""
        self.gradients.append(grad)
        self.current_worker += 1
        
        # If we have all workers' gradients, compute the majority vote
        if self.current_worker == self.param_groups[0]['num_workers']:
            self.compute_majority_vote()
            self.gradients = []
            self.current_worker = 0

    def compute_majority_vote(self):
        """Compute the majority vote of all workers' gradients"""
        if not self.gradients:
            return
            
        # Stack all gradients
        stacked_grads = torch.stack(self.gradients)
        
        # Compute the majority vote
        majority_vote = torch.sign(torch.mean(stacked_grads, dim=0))
        
        # Store the majority vote for the step
        self.majority_vote = majority_vote

    def step(self, closure=None):
        """Performs a single optimization step using the majority vote."""
        loss = None
        if closure is not None:
            loss = closure()

        if not hasattr(self, 'majority_vote'):
            return loss

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Update parameters using the majority vote
                p.data.add_(-group['lr'] * self.majority_vote)

        # Clear the majority vote after using it
        delattr(self, 'majority_vote')
        return loss
