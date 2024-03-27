from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.functional as F

class _VCReg(Function):
    @staticmethod
    def forward(ctx, input, var, cov, epsilon, demean_undo=False):
        # Batch demean the input
        mean_per_channel = input.mean(dim=(0, 1), keepdim=True)
        demeaned_input = input - mean_per_channel
        
        ctx.save_for_backward(demeaned_input)
        ctx.var = var
        ctx.cov = cov
        ctx.epsilon = epsilon
        if demean_undo:
            return input.clone()
        return demeaned_input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        demeaned_input, = ctx.saved_tensors
        var = ctx.var
        cov = ctx.cov
        epsilon = ctx.epsilon
        
        # Flatten the input to have (n, d) shape where n=batch*seq_len, d=embed
        n, seq_len, d = demeaned_input.shape
        flattened_input = demeaned_input.reshape(-1, d)
        
        # Calculate the covariance matrix
        covariance_matrix = torch.mm(flattened_input.t(), flattened_input) / (n * seq_len - 1)
        
        # Calculate the gradient
        diagonal = torch.rsqrt(covariance_matrix.diagonal() + epsilon)
        diagonal = F.threshold(diagonal, 1.0, 0.0)
        std_grad_input = diagonal * flattened_input
        cov_grad_input = torch.mm(flattened_input, covariance_matrix.fill_diagonal_(0))
        
        grad_input = grad_output \
                     - var/(d*(n*seq_len-1)) * std_grad_input.view_as(grad_output) \
                     + 4*cov/(d*(d-1)) * cov_grad_input.view_as(grad_output)
        
        return grad_input, None, None, None

class VCReg(nn.Module):
    def __init__(self, var=0.16, cov=0.01, epsilon=1e-5):
        """
        α and β serve as hyperparameters to control the strength of each regularization term.
        """
        super(VCReg, self).__init__()
        self.var = var
        self.cov = cov
        self.epsilon = epsilon

    def forward(self, input):
        return _VCReg.apply(input, self.var, self.cov, self.epsilon)
