'''
Created on Nov 23, 2025

@author: Prerana Rane with some changes by Sebastian Prepelita 
'''
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable
import tqdm
import os

from src.CNN import metrics as metrics_module

def train_epoch(model: torch.nn.Module, 
                dataloader: DataLoader, 
                lossFunction: Callable,
                optimizer: torch.optim.Optimizer, 
                device: torch.device) -> float:
    model.train() # No runtime overhead
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm.tqdm(dataloader, desc='Training', leave=False)
    for audio, pose in pbar:
        audio = audio.to(device)
        pose = pose.to(device)

        optimizer.zero_grad()
        pred_pose = model(audio)
        loss = lossFunction(pred_pose, pose)
        
        pos_loss = metrics_module.position_loss(pred_pose, pose)
        rot_loss = metrics_module.rotation_loss(pred_pose, pose)
        
        pbar.set_postfix({
            'Pos Loss': f"{pos_loss.item():.4f}",
            'Rot Loss': f"{rot_loss.item():.4f}",
            'Ratio POS/ROT': f"{pos_loss.item()/rot_loss.item():.4f}"
        })

        
        loss.backward()
        optimizer.step() # Update weights

        total_loss += loss.item()
        num_batches += 1
    avg_loss_per_batch = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss_per_batch


def train_model(model: torch.nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                lossFunction: Callable,
                optimizer: torch.optim.Optimizer,
                num_epochs: int,
                device: torch.device,
                save_dir: str = './checkpoints') -> Dict[str, list]:
    os.makedirs(save_dir, exist_ok=True)
    # Check call to getModelName(), so it crashes fast before training:
    model.getModelName()

    best_val_loss = float('inf')
    history = {
        'train_loss': np.ones(num_epochs)*-1.0, 'val_loss': np.ones(num_epochs)*-1.0,
        'val_position_mae': np.ones(num_epochs)*-1.0, 'val_rotation_mae': np.ones(num_epochs)*-1.0,
        'val_angular_error': np.ones(num_epochs)*-1.0
    }
    model.train()
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('='*60)

        train_loss = train_epoch(model = model, dataloader = train_loader, lossFunction = lossFunction, optimizer = optimizer, device = device)
        val_metrics = evaluate(model, val_loader, lossFunction, device)
        history['train_loss'][epoch] = train_loss
        history['val_loss'][epoch] = val_metrics['loss']
        history['val_position_mae'][epoch] = val_metrics['position_mae']
        history['val_rotation_mae'][epoch] = val_metrics['rotation_mae']
        history['val_angular_error'][epoch] = val_metrics['angular_error_deg']

        print(f"\nTrain Loss: {train_loss:.6f}")
        print(f"Val. Loss: {val_metrics['loss']:.6f}")
        print(f"\nPosition Metrics (Validation):")
        print(f"  Val. MSE: {val_metrics['position_mse']:.6f}")
        print(f"  Val. MAE: {val_metrics['position_mae']:.6f}")
        print(f"\nRotation Metrics (Validation):")
        print(f"  Val. Quaternion MSE: {val_metrics['rotation_mse']:.6f}")
        print(f"  Val. Quaternion MAE: {val_metrics['rotation_mae']:.6f}")
        print(f"  Val. Angular Error: {val_metrics['angular_error_deg']:.2f}\u00B0")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = os.path.join(save_dir, f'{model.getModelName()}__best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'history': history
            }, checkpoint_path)
            print(f" Saved best model (val loss: {best_val_loss:.6f})")

    # Save last model:
    checkpoint_best_path = os.path.join(save_dir, f'{model.getModelName()}__latest_trained_model.pth')
    torch.save({
        'epoch': epoch,                      # current epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
        'history': history,
    }, checkpoint_best_path)
    
    # Load and resume training (when needed):
    # checkpoint = torch.load("latest_trained_model_checkpoint.pth")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch'] + 1   # resume from next epoch
    # #last_loss = checkpoint['loss']
    # print(f"Resuming from epoch {start_epoch}")
    return history


def evaluate(model: torch.nn.Module, 
             dataloader: DataLoader, 
             lossFunction: Callable,
             device: torch.device,
             evalTest = False) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    all_metrics = {
        'position_mse': np.full(len(dataloader), np.nan, dtype = np.float64), 'position_mae': np.full(len(dataloader), np.nan, dtype = np.float64),
        'rotation_mse': np.full(len(dataloader), np.nan, dtype = np.float64), 'rotation_mae': np.full(len(dataloader), np.nan, dtype = np.float64),
        'angular_error_deg': np.full(len(dataloader), np.nan, dtype = np.float64)
    }
    if evalTest:
        desc_ = 'Testing'
    else:
        desc_ = 'Evaluating'
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc=desc_, leave=False)
        for idx, (audio, pose) in enumerate(pbar):
            audio = audio.to(device)
            pose = pose.to(device)

            pred_pose = model(audio)
            loss = lossFunction(pred_pose, pose)
            
            pos_loss = metrics_module.position_loss(pred_pose, pose)
            rot_loss = metrics_module.rotation_loss(pred_pose, pose)
            
            pbar.set_postfix({
                'Pos Loss': f"{pos_loss.item():.4f}",
                'Rot Loss': f"{rot_loss.item():.4f}",
                'Ratio POS/ROT': f"{pos_loss.item()/rot_loss.item():.4f}"
            })
            total_loss += loss.item()
            metrics = metrics_module.compute_metrics(pred_pose, pose)
            for key, value in metrics.items():
                all_metrics[key][idx] = value

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics

def evaluate_per_samples(model: torch.nn.Module, 
             dataloader: DataLoader, 
             lossFunction: Callable,
             device: torch.device,
             evalTest = False) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    num_samples = len(dataloader.dataset)  # total samples, not batches
    
    all_metrics = {
        'position_mse': np.full(num_samples, np.nan, dtype = np.float64), 'position_mae': np.full(num_samples, np.nan, dtype = np.float64),
        'rotation_mse': np.full(num_samples, np.nan, dtype = np.float64), 'rotation_mae': np.full(num_samples, np.nan, dtype = np.float64),
        'angular_error_deg': np.full(num_samples, np.nan, dtype = np.float64)
    }
    if evalTest:
        desc_ = 'Testing [per sample]'
    else:
        desc_ = 'Evaluating [per sample]'
    start_idx = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc=desc_, leave=False)
        for audio_batch, pose_batch in pbar:
            batch_size = audio_batch.size(0)
            end_idx = start_idx + batch_size
            
            audio = audio_batch.to(device)
            pose = pose_batch.to(device)

            pred_pose = model(audio)
            loss = lossFunction(pred_pose, pose)
            
            pos_loss = metrics_module.position_loss(pred_pose, pose)
            rot_loss = metrics_module.rotation_loss(pred_pose, pose)
            
            pbar.set_postfix({
                'Pos Loss': f"{pos_loss.item():.4f}",
                'Rot Loss': f"{rot_loss.item():.4f}",
                'Ratio POS/ROT': f"{pos_loss.item()/rot_loss.item():.4f}"
            })
            total_loss += loss.item()
            per_sample_metrics = metrics_module.compute_metrics_per_sample(pred_pose, pose)
            for key, values in per_sample_metrics.items():
                all_metrics[key][start_idx:end_idx] = values
            start_idx = end_idx

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics

def load_checkpoint(model: torch.nn.Module, 
                   checkpoint_path: str,
                   device: torch.device,
                   optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch_checkpoint = checkpoint['epoch']
    # history_checkpoint = checkpoint['history']
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f" Loaded checkpoint from epoch {epoch_checkpoint + 1}")
    print(f" Validation loss: {checkpoint['val_loss']:.6f}")
    
    return checkpoint