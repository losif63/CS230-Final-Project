import torch
import torch.nn.functional as F
from typing import Dict
import numpy as np

def position_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    position_loss = F.mse_loss(pred[:, :3], target[:, :3]) # Position loss (MSE on x, y, z)
    return position_loss

def rotation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #Rotation loss (quaternion) - handles q and -q representing same rotation
    pred_quat = pred[:, 3:]
    target_quat = target[:, 3:]

    #q and -q losses
    quat_loss_pos = F.mse_loss(pred_quat, target_quat)
    quat_loss_neg = F.mse_loss(pred_quat, -target_quat)
    rotation_loss = torch.min(quat_loss_pos, quat_loss_neg)
    
    return rotation_loss

def pose_6dof_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #Combined loss for position and rotation.
    return position_loss(pred, target) + rotation_loss(pred, target)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    # Returns the metrics per batch:
    
    # Compute MSE and MAE for position and rotation
    # Position metrics
    position_error = pred[:, :3] - target[:, :3]
    position_mse = torch.sqrt((position_error ** 2).sum(dim=1)).mean().item() # Euclidean
    position_mae = position_error.abs().mean().item()

    # Rotation metrics (quaternion MSE/MAE)
    pred_quat = pred[:, 3:]
    target_quat = target[:, 3:]

    # Handle q/-q ambiguity for rotation error
    quat_error_pos = (pred_quat - target_quat).abs()
    quat_error_neg = (pred_quat + target_quat).abs()
    quat_error = torch.min(quat_error_pos, quat_error_neg)

    rotation_mse = (quat_error ** 2).mean().item()
    rotation_mae = quat_error.mean().item()

    # Angular error in degrees
    pred_quat_norm = F.normalize(pred_quat, p=2, dim=1)
    target_quat_norm = F.normalize(target_quat, p=2, dim=1)

    dot_product = torch.sum(pred_quat_norm * target_quat_norm, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    angular_error_rad = 2 * torch.acos(torch.abs(dot_product))
    angular_error_deg = torch.rad2deg(angular_error_rad).mean().item()

    return {
        'position_mse': position_mse,
        'position_mae': position_mae,
        'rotation_mse': rotation_mse,
        'rotation_mae': rotation_mae,
        'angular_error_deg': angular_error_deg
    }


def compute_metrics_per_sample(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Compute per-sample metrics for position and rotation.
    
    Same as compute_metrics, but returns the value per each sample in the batch.
    
    Returns NumPy arrays of shape [batch_size].
    """
    # --- Position metrics ---
    # Compute MSE and MAE for position and rotation
    # Position metrics
    position_error = pred[:, :3] - target[:, :3]
    # Euclidean distance per sample
    position_mse = torch.sqrt((position_error ** 2).sum(dim=1)) # shape [batch_size]
    # MAE per sample (mean over 3 coords, not batch)
    position_mae = position_error.abs().mean(dim=1)         # shape [batch_size]

    # --- Rotation metrics (quaternion) --- (MSE/MAE)
    pred_quat = pred[:, 3:]
    target_quat = target[:, 3:]

    # Handle q/-q ambiguity for rotation error
    quat_error_pos = (pred_quat - target_quat).abs()
    quat_error_neg = (pred_quat + target_quat).abs()
    quat_error = torch.min(quat_error_pos, quat_error_neg)

    rotation_mse = (quat_error ** 2).mean(dim=1) # per sample, shape [batch_size]
    rotation_mae = quat_error.mean(dim=1) # per sample, shape [batch_size]

    # Angular error in degrees
    pred_quat_norm = F.normalize(pred_quat, p=2, dim=1)
    target_quat_norm = F.normalize(target_quat, p=2, dim=1)

    dot_product = torch.sum(pred_quat_norm * target_quat_norm, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    angular_error_rad = 2 * torch.acos(torch.abs(dot_product))
    # Angular error per sample:
    angular_error_deg = torch.rad2deg(angular_error_rad) # shape [batch_size]
    
    # --- Return as NumPy arrays ---
    return {
        'position_mse': position_mse.cpu().numpy(),
        'position_mae': position_mae.cpu().numpy(),
        'rotation_mse': rotation_mse.cpu().numpy(),
        'rotation_mae': rotation_mae.cpu().numpy(),
        'angular_error_deg': angular_error_deg.cpu().numpy()
    }