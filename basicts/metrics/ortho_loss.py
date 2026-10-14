import numpy as np
import torch
import torch.nn.functional as F


def masked_mae_with_ortho(prediction: torch.Tensor, target: torch.Tensor, 
                          ortho_loss: torch.Tensor = None, 
                          ortho_weight: float = None,
                          null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Absolute Error (MAE) combined with mode orthogonality loss.
    
    This loss function encourages different modes in the multi-mode ODE decoder to learn
    orthogonal dynamics, promoting mode diversity.5

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        ortho_loss (torch.Tensor, optional): The orthogonality loss from the decoder. Default is None.
        ortho_weight (float, optional): Weight for the orthogonality regularization term. Default is 0.1.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Default is `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the combined loss.

    """
    # Calculate masked MAE
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    mae_loss = torch.abs(prediction - target)
    mae_loss = mae_loss * mask  # Apply the mask to the loss
    mae_loss = torch.nan_to_num(mae_loss)  # Replace any NaNs in the loss with zero
    mae_loss = torch.mean(mae_loss)
    
    # Add orthogonality regularization if provided
    if ortho_loss is not None and ortho_weight is not None:
        total_loss = mae_loss + ortho_weight * ortho_loss
    else:
        total_loss = mae_loss
    
    return total_loss