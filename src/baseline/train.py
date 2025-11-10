import os
from typing import Dict, Any, Callable
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from metrics import compute_metrics


def train_epoch(model: torch.nn.Module, 
                dataloader: DataLoader, 
                criterion: Callable,
                optimizer: torch.optim.Optimizer, 
                device: torch.device) -> float:
    model.train()

    total_loss = 0.0
    num_batches = 0

    for audio, pose in tqdm(dataloader, desc='Training', leave=False):
        audio = audio.to(device)
        pose = pose.to(device)

        optimizer.zero_grad()
        pred_pose = model(audio)
        loss = criterion(pred_pose, pose)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model: torch.nn.Module, 
             dataloader: DataLoader, 
             criterion: Callable,
             device: torch.device) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    all_metrics = {
        'position_mse': [], 'position_mae': [],
        'rotation_mse': [], 'rotation_mae': [],
        'angular_error_deg': []
    }

    with torch.no_grad():
        for audio, pose in tqdm(dataloader, desc='Evaluating', leave=False):
            audio = audio.to(device)
            pose = pose.to(device)

            pred_pose = model(audio)
            loss = criterion(pred_pose, pose)
            total_loss += loss.item()
            metrics = compute_metrics(pred_pose, pose)
            for key, value in metrics.items():
                all_metrics[key].append(value)

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics


def train_model(model: torch.nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: Callable,
                optimizer: torch.optim.Optimizer,
                num_epochs: int,
                device: torch.device,
                save_dir: str = './checkpoints') -> Dict[str, list]:
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'val_position_mae': [], 'val_rotation_mae': [],
        'val_angular_error': []
    }

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('='*60)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_position_mae'].append(val_metrics['position_mae'])
        history['val_rotation_mae'].append(val_metrics['rotation_mae'])
        history['val_angular_error'].append(val_metrics['angular_error_deg'])

        print(f"\nTrain Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_metrics['loss']:.6f}")
        print(f"\nPosition Metrics:")
        print(f"  MSE: {val_metrics['position_mse']:.6f}")
        print(f"  MAE: {val_metrics['position_mae']:.6f}")
        print(f"\nRotation Metrics:")
        print(f"  Quaternion MSE: {val_metrics['rotation_mse']:.6f}")
        print(f"  Quaternion MAE: {val_metrics['rotation_mae']:.6f}")
        print(f"  Angular Error: {val_metrics['angular_error_deg']:.2f}°")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'history': history
            }, checkpoint_path)
            print(f" Saved best model (val loss: {best_val_loss:.6f})")

    return history


def load_checkpoint(model: torch.nn.Module, 
                   checkpoint_path: str,
                   device: torch.device,
                   optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f" Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    print(f" Validation loss: {checkpoint['val_loss']:.6f}")
    
    return checkpoint
