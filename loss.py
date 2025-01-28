import torch 
import torch.nn as nn 
import torch.nn.functional as F

def covariance_loss(x):
    """
    Parameters:
        x (Tensor): The input tensor of shape (batch_size, ..., features_dimension).

    Returns:
        Tensor: The computed covariance loss.
    """
    # Center the tensor by subtracting its mean
    x_centered = x - x.mean(dim=0)
    batch_size = x_centered.size(0)
    features_dim = x_centered.size(-1)
    non_diag_mask = ~torch.eye(features_dim, device=x.device, dtype=torch.bool)
    covariance_matrix = torch.einsum("b...i,b...j->...ij", x_centered, x_centered) / (batch_size - 1)
    loss = covariance_matrix[..., non_diag_mask].pow(2).sum(-1) / features_dim
    return loss.mean()

class VICLoss(nn.Module):


    def __init__(
        self,
        invariance_term: float = 25.0,
        variance_term: float = 25.0,
        covariance_term: float = 1.0,
        eps=1e-5,
    ):
        super(VICLoss, self).__init__()

        self.invariance_term = invariance_term
        self.variance_term = variance_term
        self.covariance_term = covariance_term
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        invariance_loss =  F.mse_loss(x, y)

        std_x = torch.sqrt(x.var(dim=0) + self.eps)
        x_loss = torch.mean(F.relu(1.0 - std_x))

        std_y= torch.sqrt(y.var(dim=0) + self.eps)
        y_loss = torch.mean(F.relu(1.0 - std_y))

        variance_loss = (x_loss + y_loss) / 2.0

    
        covariance_loss = covariance_loss(x=x) + covariance_loss(x=y)
        loss = self.invariance_term * invariance_loss + self.variance_term * variance_loss + self.covariance_term * covariance_loss
        return loss, (invariance_loss, variance_loss, covariance_loss)



