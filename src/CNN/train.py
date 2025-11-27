'''
Created on Nov 23, 2025

    Part of training of CNN models, CS230 project, fall 2025. 

@author: Prerana Rane with some changes by Sebastian Prepelita
'''
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable, Tuple, Optional
import tqdm
import os
import sys
import logging

from src.CNN import metrics as metrics_module


def is_interactive_environment() -> bool:
    """
    Detect if we're running in an interactive environment (local terminal)
    or a non-interactive environment (SLURM, batch job, etc.).

    Returns
    -------
    bool
        True if interactive (show TQDM progress bars), False if non-interactive (disable TQDM)
    """
    if not sys.stderr.isatty():
        return False

    if any(var in os.environ for var in ['SLURM_JOB_ID', 'SLURM_JOBID', 'PBS_JOBID', 'LSB_JOBID']):
        return False

    return True


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that works with tqdm progress bars."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
        except Exception:
            self.handleError(record)


def create_tqdm_callback(logger: Optional[logging.Logger], desc: str, total: int):
    """
    Create a tqdm callback that logs progress at 25% intervals.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger instance
    desc : str
        Description for the progress bar
    total : int
        Total number of items

    Returns
    -------
    function
        Callback function for tqdm
    """
    if logger is None:
        return None

    milestones = {int(total * 0.25), int(total * 0.5), int(total * 0.75), total}
    logged_milestones = set()

    def callback(pbar):
        current = pbar.n
        for milestone in milestones:
            if current >= milestone and milestone not in logged_milestones:
                percentage = (milestone / total) * 100
                logger.info(f"{desc}: {percentage:.0f}% complete ({milestone}/{total})")
                logged_milestones.add(milestone)

    return callback

def train_epoch(epoch: int,
                model: torch.nn.Module,
                dataloader: DataLoader,
                lossFunction: Callable,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                logger: Optional[logging.Logger] = None) -> float:
    model.train() # No runtime overhead
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm.tqdm(dataloader, desc='Training', leave=False, disable=not is_interactive_environment())
    # Set up progress logging milestones
    total_batches = len(dataloader)
    milestones = {int(total_batches * 0.25), int(total_batches * 0.5),
                  int(total_batches * 0.75), total_batches}
    logged_milestones = set()
    for batch_idx, (audio, pose) in enumerate(pbar):
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

        # Log progress at milestones
        if logger:
            for milestone in milestones:
                if batch_idx >= milestone and milestone not in logged_milestones:
                    percentage = (milestone / total_batches) * 100
                    logger.debug(f"'Training EPOCH {epoch}': {percentage:.0f}% complete ({milestone}/{total_batches} batches)")
                    logged_milestones.add(milestone)
    avg_loss_per_batch = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss_per_batch


def train_model(model: torch.nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                lossFunction: Callable,
                optimizer: torch.optim.Optimizer,
                num_epochs: int,
                device: torch.device,
                save_dir: str = './checkpoints',
                logger: Optional[logging.Logger] = None) -> Dict[str, list]:
    """
    Train a model with logging support.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger instance for logging training progress
    """
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

    # Set up tqdm logging callback
    if logger:
        # Add tqdm-compatible handler temporarily
        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setFormatter(logger.handlers[0].formatter if logger.handlers else None)
        logger.addHandler(tqdm_handler)

    for epoch in range(num_epochs):
        epoch_header = f"\t\tEpoch {epoch+1}/{num_epochs}, on device '{device}'"
        if logger:
            logger.debug(epoch_header)

        train_loss = train_epoch(epoch = epoch, model = model, dataloader = train_loader, lossFunction = lossFunction, optimizer = optimizer, device = device, logger=logger)
        val_metrics = evaluate(model, val_loader, lossFunction, device, logger=logger)
        history['train_loss'][epoch] = train_loss
        history['val_loss'][epoch] = val_metrics['loss']
        history['val_position_mae'][epoch] = val_metrics['position_mae']
        history['val_rotation_mae'][epoch] = val_metrics['rotation_mae']
        history['val_angular_error'][epoch] = val_metrics['angular_error_deg']

        results_msg = (
            f"\t\tTrain Loss: {train_loss:.6f}\n"
            f"\t\tVal. Loss: {val_metrics['loss']:.6f}\n"
            f"\n"
            f"\t\tPosition Metrics (Validation):\n"
            f"\t\t  Val. MSE: {val_metrics['position_mse']:.6f}\n"
            f"\t\t  Val. MAE: {val_metrics['position_mae']:.6f}\n"
            f"\n"
            f"\t\tRotation Metrics (Validation):\n"
            f"\t\t  Val. Quaternion MSE: {val_metrics['rotation_mse']:.6f}\n"
            f"\t\t  Val. Quaternion MAE: {val_metrics['rotation_mae']:.6f}\n"
            f"\t\t  Val. Angular Error: {val_metrics['angular_error_deg']:.2f}°"
        )

        if logger:
            logger.debug(results_msg)

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

            best_model_msg = f"    Saved best model (val loss: {best_val_loss:.6f})"
            if logger:
                logger.debug(best_model_msg)

    # Save last model:
    checkpoint_best_path = os.path.join(save_dir, f'{model.getModelName()}__latest_trained_model.pth')
    torch.save({
        'epoch': epoch,                      # current epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
        'history': history,
    }, checkpoint_best_path)

    # Remove tqdm handler
    if logger:
        logger.removeHandler(tqdm_handler)

    return history


def evaluate(model: torch.nn.Module,
             dataloader: DataLoader,
             lossFunction: Callable,
             device: torch.device,
             evalTest = False,
             logger: Optional[logging.Logger] = None) -> Dict[str, float]:
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
    # Set up progress logging milestones
    total_batches = len(dataloader)
    milestones = {int(total_batches * 0.33), int(total_batches * 0.66),
                  total_batches}
    logged_milestones = set()

    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc=desc_, leave=False, disable=not is_interactive_environment())
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

            # Log progress at milestones
            if logger:
                for milestone in milestones:
                    if idx >= milestone and milestone not in logged_milestones:
                        percentage = (milestone / total_batches) * 100
                        logger.debug(f"{desc_}: {percentage:.0f}% complete ({milestone}/{total_batches} batches)")
                        logged_milestones.add(milestone)

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics

def evaluate_per_samples(model: torch.nn.Module,
              dataloader: DataLoader,
              lossFunction: Callable,
              device: torch.device,
              evalTest = False,
              logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Evaluate model per sample with logging support.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger instance for logging evaluation progress
    """
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

    # Set up progress logging milestones
    total_batches = len(dataloader)
    milestones = {int(total_batches * 0.25), int(total_batches * 0.5),
                  int(total_batches * 0.75), total_batches}
    logged_milestones = set()

    start_idx = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc=desc_, leave=False, disable=not is_interactive_environment())
        for batch_idx, (audio_batch, pose_batch) in enumerate(pbar, 1):
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

            # Log progress at milestones
            if logger:
                for milestone in milestones:
                    if batch_idx >= milestone and milestone not in logged_milestones:
                        percentage = (milestone / total_batches) * 100
                        logger.debug(f"{desc_}: {percentage:.0f}% complete ({milestone}/{total_batches} batches)")
                        logged_milestones.add(milestone)

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics, all_metrics

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
