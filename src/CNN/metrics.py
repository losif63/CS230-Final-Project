import torch
import torch.nn.functional as F
from typing import Dict
import numpy as np


def position_loss(
    pred: torch.Tensor, target: torch.Tensor, output_dim: int = 7
) -> torch.Tensor:
    if output_dim == 3 or output_dim == 7:
        position_loss = F.mse_loss(pred[:, :3], target[:, :3])
    else:
        position_loss = torch.tensor(0.0, device=pred.device)
    return position_loss


def rotation_loss(
    pred: torch.Tensor, target: torch.Tensor, output_dim: int = 7
) -> torch.Tensor:
    if output_dim == 4:
        pred_quat = pred
        target_quat = target
        quat_loss_pos = F.mse_loss(pred_quat, target_quat)
        quat_loss_neg = F.mse_loss(pred_quat, -target_quat)
        rotation_loss = torch.min(quat_loss_pos, quat_loss_neg)
    elif output_dim == 7:
        pred_quat = pred[:, 3:]
        target_quat = target[:, 3:]
        quat_loss_pos = F.mse_loss(pred_quat, target_quat)
        quat_loss_neg = F.mse_loss(pred_quat, -target_quat)
        rotation_loss = torch.min(quat_loss_pos, quat_loss_neg)
    else:
        rotation_loss = torch.tensor(0.0, device=pred.device)
    return rotation_loss


def pose_6dof_loss(
    pred: torch.Tensor, target: torch.Tensor, output_dim: int = 7
) -> torch.Tensor:
    return position_loss(pred, target, output_dim) + rotation_loss(
        pred, target, output_dim
    )


def compute_metrics(
    pred: torch.Tensor, target: torch.Tensor, output_dim: int = 7
) -> Dict[str, float]:
    if output_dim == 3:
        position_error = pred - target
        position_mse = torch.sqrt((position_error**2).sum(dim=1)).mean().item()
        position_mae = position_error.abs().mean().item()
        rotation_mse = 0.0
        rotation_mae = 0.0
        angular_error_deg = 0.0
    elif output_dim == 4:
        position_mse = 0.0
        position_mae = 0.0
        pred_quat = pred
        target_quat = target
        quat_error_pos = (pred_quat - target_quat).abs()
        quat_error_neg = (pred_quat + target_quat).abs()
        quat_error = torch.min(quat_error_pos, quat_error_neg)
        rotation_mse = (quat_error**2).mean().item()
        rotation_mae = quat_error.mean().item()
        pred_quat_norm = F.normalize(pred_quat, p=2, dim=1)
        target_quat_norm = F.normalize(target_quat, p=2, dim=1)
        dot_product = torch.sum(pred_quat_norm * target_quat_norm, dim=1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angular_error_rad = 2 * torch.acos(torch.abs(dot_product))
        angular_error_deg = torch.rad2deg(angular_error_rad).mean().item()
    else:
        position_error = pred[:, :3] - target[:, :3]
        position_mse = torch.sqrt((position_error**2).sum(dim=1)).mean().item()
        position_mae = position_error.abs().mean().item()
        pred_quat = pred[:, 3:]
        target_quat = target[:, 3:]
        quat_error_pos = (pred_quat - target_quat).abs()
        quat_error_neg = (pred_quat + target_quat).abs()
        quat_error = torch.min(quat_error_pos, quat_error_neg)
        rotation_mse = (quat_error**2).mean().item()
        rotation_mae = quat_error.mean().item()
        pred_quat_norm = F.normalize(pred_quat, p=2, dim=1)
        target_quat_norm = F.normalize(target_quat, p=2, dim=1)
        dot_product = torch.sum(pred_quat_norm * target_quat_norm, dim=1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angular_error_rad = 2 * torch.acos(torch.abs(dot_product))
        angular_error_deg = torch.rad2deg(angular_error_rad).mean().item()

    return {
        "position_mse": position_mse,
        "position_mae": position_mae,
        "rotation_mse": rotation_mse,
        "rotation_mae": rotation_mae,
        "angular_error_deg": angular_error_deg,
    }


def compute_metrics_per_sample(
    pred: torch.Tensor, target: torch.Tensor, output_dim: int = 7
) -> Dict[str, np.ndarray]:
    """
    Compute per-sample metrics for position and rotation.

    Same as compute_metrics, but returns the value per each sample in the batch.

    Returns NumPy arrays of shape [batch_size].
    """
    batch_size = pred.shape[0]

    if output_dim == 3:
        position_error = pred - target
        position_mse = torch.sqrt((position_error**2).sum(dim=1))
        position_mae = position_error.abs().mean(dim=1)
        rotation_mse = torch.zeros(batch_size, device=pred.device)
        rotation_mae = torch.zeros(batch_size, device=pred.device)
        angular_error_deg = torch.zeros(batch_size, device=pred.device)
    elif output_dim == 4:
        position_mse = torch.zeros(batch_size, device=pred.device)
        position_mae = torch.zeros(batch_size, device=pred.device)
        pred_quat = pred
        target_quat = target
        quat_error_pos = (pred_quat - target_quat).abs()
        quat_error_neg = (pred_quat + target_quat).abs()
        quat_error = torch.min(quat_error_pos, quat_error_neg)
        rotation_mse = (quat_error**2).mean(dim=1)
        rotation_mae = quat_error.mean(dim=1)
        pred_quat_norm = F.normalize(pred_quat, p=2, dim=1)
        target_quat_norm = F.normalize(target_quat, p=2, dim=1)
        dot_product = torch.sum(pred_quat_norm * target_quat_norm, dim=1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angular_error_rad = 2 * torch.acos(torch.abs(dot_product))
        angular_error_deg = torch.rad2deg(angular_error_rad)
    else:
        position_error = pred[:, :3] - target[:, :3]
        position_mse = torch.sqrt((position_error**2).sum(dim=1))
        position_mae = position_error.abs().mean(dim=1)
        pred_quat = pred[:, 3:]
        target_quat = target[:, 3:]
        quat_error_pos = (pred_quat - target_quat).abs()
        quat_error_neg = (pred_quat + target_quat).abs()
        quat_error = torch.min(quat_error_pos, quat_error_neg)
        rotation_mse = (quat_error**2).mean(dim=1)
        rotation_mae = quat_error.mean(dim=1)
        pred_quat_norm = F.normalize(pred_quat, p=2, dim=1)
        target_quat_norm = F.normalize(target_quat, p=2, dim=1)
        dot_product = torch.sum(pred_quat_norm * target_quat_norm, dim=1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angular_error_rad = 2 * torch.acos(torch.abs(dot_product))
        angular_error_deg = torch.rad2deg(angular_error_rad)

    return {
        "position_mse": position_mse.cpu().numpy(),
        "position_mae": position_mae.cpu().numpy(),
        "rotation_mse": rotation_mse.cpu().numpy(),
        "rotation_mae": rotation_mae.cpu().numpy(),
        "angular_error_deg": angular_error_deg.cpu().numpy(),
    }
