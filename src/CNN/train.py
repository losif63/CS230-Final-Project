"""
Created on Nov 23, 2025

    Part of training of CNN models, CS230 project, fall 2025.

@author: Prerana Rane with some changes by Sebastian Prepelita
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable, Tuple, Optional
import tqdm
import os
import sys
import logging

from src.CNN import metrics as metrics_module
from src.CNN.dataset import apply_spectogram_batch_gpu


def _debug_save_spectrogram_visualizations(
    audio_batch_device: torch.Tensor,
    train_idx: int = 0,
    str_compute: str = "[UNKNOWN]",
    save_dir: str = "."
) -> None:
    """
    Debug function to save spectrogram visualizations for CPU vs GPU comparison.
    
    This saves one PNG file per channel with the spectrogram visualization and
    prints statistics for the first channel.
    
    Parameters
    ----------
    audio_batch_device : torch.Tensor
        Batch of audio/spectrogram data of shape (batch_size, n_channels, freq_bins, time_bins)
        Should already be on the target device (GPU or CPU)
    train_idx : int, default=0
        Index of the sample in the batch to visualize
    str_compute : str, default="[UNKNOWN]"
        String indicating computation mode (e.g., "[GPU]" or "[CPU]")
        Used in filenames and titles
    save_dir : str, default="."
        Directory where PNG files should be saved (defaults to current directory)
        
    Notes
    -----
    This function is intended for debugging and validation purposes. It creates
    one visualization per channel, which can be used to visually compare CPU vs GPU
    spectrogram computation outputs.
    
    Files are saved as: `spectrogram_{compute_mode}_channel_{ch_idx}.png`
    
    Example
    -------
    >>> # After computing spectrograms on GPU
    >>> _debug_save_spectrogram_visualizations(audio, train_idx=0, str_compute="[GPU]")
    >>> # After computing spectrograms on CPU
    >>> _debug_save_spectrogram_visualizations(audio, train_idx=0, str_compute="[CPU]")
    >>> # Compare the generated PNG files
    """
    import matplotlib.pyplot as plt
    # Get the spectrogram for the fixed sample
    spec_sample = audio_batch_device[train_idx].cpu().numpy()  # Shape: (n_channels, freq_bins, time_bins)
    n_channels, n_freq_bins, n_time_bins = spec_sample.shape
    
    # Sampling frequency
    fs = 48000.0
    # Compute frequency axis (only positive frequencies if spectrogram is magnitude)
    freqs = np.fft.fftfreq(n_freq_bins * 2, d=1/fs)[:n_freq_bins]
    freqs_kHz = freqs*1e-3
    
    print(f"\n{'='*80}")
    print(f"SAVING SPECTROGRAM VISUALIZATIONS {str_compute}")
    print(f"  Sample shape: {spec_sample.shape}")
    print(f"  Saving to: {save_dir}")
    
    # Loop through all channels and save individual spectrograms
    for ch_idx in range(n_channels):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot spectrogram
        im = ax.imshow(
            spec_sample[ch_idx, :, :],
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest',
            extent=[0, n_time_bins, freqs_kHz[0], freqs_kHz[-1]]# map y-axis to frequency
        )
        
        ax.set_title(f'{str_compute} Spectrogram - Channel {ch_idx}', fontsize=14)
        ax.set_xlabel('Time Bins', fontsize=12)
        ax.set_ylabel('Frequency [kHz]', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitude', fontsize=12)
        
        # Save figure
        compute_mode = str_compute.replace('[', '').replace(']', '')
        filename = os.path.join(save_dir, f"spectrogram_{compute_mode}_channel_{ch_idx}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"    Saved: {filename}")
    
    print(f"  Statistics for channel 0 {str_compute}:")
    print(f"    Min {str_compute}: {spec_sample[0].min():.4f}")
    print(f"    Max {str_compute}: {spec_sample[0].max():.4f}")
    print(f"    Mean {str_compute}: {spec_sample[0].mean():.4f}")
    print(f"    Std {str_compute}: {spec_sample[0].std():.4f}")
    print(f"{'='*80}\n")


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

    if any(
        var in os.environ
        for var in ["SLURM_JOB_ID", "SLURM_JOBID", "PBS_JOBID", "LSB_JOBID"]
    ):
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


def train_epoch(
    epoch: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    lossFunction: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    gpu_stft_params: Optional[dict] = None,
) -> float:
    """
    Train model for one epoch with optional GPU-accelerated spectrogram computation.
    
    This function performs one full pass through the training dataset, computing
    forward pass, loss, and backpropagation. It supports both pre-computed 
    spectrograms (CPU) and on-the-fly GPU spectrogram computation for maximum
    performance.
    
    Parameters
    ----------
    epoch : int
        Current epoch number (0-indexed), used for logging
    model : torch.nn.Module
        Neural network model to train. Must have `output_dim` attribute
        indicating output dimensionality (3, 4, or 7)
    dataloader : DataLoader
        Training data loader providing (audio, pose) batches
    lossFunction : Callable
        Loss function with signature: loss_fn(pred_pose, true_pose, output_dim) -> Tensor
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters (e.g., Adam, SGD)
    device : torch.device
        Target device for computation (e.g., 'cuda:0', 'cpu')
    logger : logging.Logger, optional
        Logger for progress tracking. If None, no logging is performed
    gpu_stft_params : dict, optional
        Parameters for GPU-accelerated spectrogram computation. If None, assumes
        data loader returns pre-computed spectrograms. If provided, raw audio
        from dataloader will be converted to spectrograms on GPU.
        
        Required keys:
            'window' : torch.Tensor
                Pre-allocated window tensor on GPU, shape (n_window,)
            'hop_size' : int
                STFT hop size in samples (e.g., 2)
            'mfft' : int
                FFT size, computed using _compute_mfft(samples_per_frame, hop_size)
                This should match the mfft used in CPU spectrogram computation
        
        Optional keys:
            'pos_encoding' : torch.Tensor or None
                Pre-computed positional encoding on GPU, shape (1, freq_bins, time_bins)
    
    Returns
    -------
    float
        Average loss per batch for the epoch
        
    Notes
    -----
    **GPU Spectrogram Mode**:
    When `gpu_stft_params` is provided, this function expects the dataloader to
    return raw audio tensors (batch, channels, samples) instead of pre-computed
    spectrograms. The STFT is then computed on GPU using PyTorch's optimized
    implementation, providing ~30x speedup over CPU scipy computation.
    
    **Progress Logging**:
    Progress is logged at 25%, 50%, 75%, and 100% completion if logger is provided.
    A tqdm progress bar is displayed in interactive environments.
    
    **Performance Considerations**:
    - GPU spectrogram computation adds <1ms per batch (negligible overhead)
    - Pre-computing spectrograms on CPU takes ~30ms per sample (significant bottleneck)
    - GPU mode enables higher GPU utilization and faster training
    
    See Also
    --------
    train_model : Full training loop across multiple epochs
    evaluate : Model evaluation without gradient computation
    apply_spectogram_batch_gpu : GPU spectrogram computation function
    """
    model.train()  # No runtime overhead
    total_loss = 0.0
    num_batches = 0
    output_dim = model.output_dim
    pbar = tqdm.tqdm(
        dataloader,
        desc="Training",
        leave=False,
        disable=not is_interactive_environment(),
    )
    # Set up progress logging milestones
    total_batches = len(dataloader)
    milestones = {
        int(total_batches * 0.25),
        int(total_batches * 0.5),
        int(total_batches * 0.75),
        total_batches,
    }
    logged_milestones = set()
    for batch_idx, (audio, pose) in enumerate(pbar):
        audio = audio.to(device)
        pose = pose.to(device)
        
        # Compute spectrograms on GPU if enabled
        if gpu_stft_params is not None:
            audio = apply_spectogram_batch_gpu(
                audio,
                window=gpu_stft_params['window'],
                hop_size=gpu_stft_params['hop_size'],
                mfft=gpu_stft_params['mfft'],
                pos_encoding_tensor=gpu_stft_params.get('pos_encoding', None)
            )
            # Apply window normalization and 100x scaling in one operation (optimization)
            # scipy's ShortTimeFFT divides by window sum, then we multiply by 100
            audio = audio * (100.0 / gpu_stft_params['window_sum'])
        
        # ============================================================================
        # DEBUG: Visualize spectrograms for CPU vs GPU comparison
        # ============================================================================
        # str_compute = "[GPU]" if gpu_stft_params is not None else "[CPU]"
        # if batch_idx == 0 and epoch == 0:  # Only do this once at the start
        #     _debug_save_spectrogram_visualizations(
        #         audio_batch_device=audio,
        #         train_idx=0,
        #         str_compute=str_compute,
        #         save_dir="."
        #     )
        # print("DEBUGGED SPECTOGRAMS!")
        # sys.exit()
        # ============================================================================

        optimizer.zero_grad()
        pred_pose = model(audio)
        loss = lossFunction(pred_pose, pose, output_dim)

        pos_loss = metrics_module.position_loss(pred_pose, pose, output_dim)
        rot_loss = metrics_module.rotation_loss(pred_pose, pose, output_dim)

        if rot_loss.item() > 0.0:
            ratio_text = f"{pos_loss.item()/rot_loss.item():.4f}"
        else:
            ratio_text = "N/A"

        pbar.set_postfix(
            {
                "Pos Loss": f"{pos_loss.item():.4f}",
                "Rot Loss": f"{rot_loss.item():.4f}",
                "Ratio POS/ROT": ratio_text,
            }
        )

        loss.backward()
        optimizer.step()  # Update weights

        total_loss += loss.item()
        num_batches += 1

        # Log progress at milestones
        if logger:
            for milestone in milestones:
                if batch_idx >= milestone and milestone not in logged_milestones:
                    percentage = (milestone / total_batches) * 100
                    logger.debug(
                        f"'Training EPOCH {epoch}': {percentage:.0f}% complete ({milestone}/{total_batches} batches)"
                    )
                    logged_milestones.add(milestone)
        
    avg_loss_per_batch = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss_per_batch


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lossFunction: Callable,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: str = "./checkpoints",
    logger: Optional[logging.Logger] = None,
    gpu_stft_params: Optional[dict] = None,
) -> Dict[str, list]:
    """
    Train a neural network model for audio-based head pose estimation with automatic checkpointing.
    
    This is the main training loop that orchestrates the entire training process, including:
    - Running training and validation epochs
    - Computing and tracking multiple metrics (position MAE, rotation MAE, angular error)
    - Automatic model checkpointing (best model based on validation loss)
    - Comprehensive logging and progress tracking
    - Support for GPU-accelerated spectrogram computation
    
    The function automatically saves two checkpoints:
    1. Best model: Saved whenever validation loss improves
    2. Latest model: Saved at the end of training (final epoch)
    
    Parameters
    ----------
    model : torch.nn.Module
        Neural network model to train. Must have:
        - `output_dim` attribute indicating output dimensionality (3, 4, or 7)
        - `getModelName()` method for generating checkpoint filenames
    train_loader : DataLoader
        Training data loader providing (audio, pose) batches
    val_loader : DataLoader
        Validation data loader for model evaluation after each epoch
    lossFunction : Callable
        Loss function with signature: loss_fn(pred_pose, true_pose, output_dim) -> Tensor
        Used for both training and validation
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters (e.g., Adam, SGD)
        State is saved in checkpoints for training resumption
    num_epochs : int
        Number of training epochs to run
    device : torch.device
        Target device for computation (e.g., 'cuda:0', 'cpu')
        All data and models are moved to this device
    save_dir : str, default="./checkpoints"
        Directory where model checkpoints will be saved
        Created automatically if it doesn't exist
    logger : logging.Logger, optional
        Logger instance for detailed progress tracking. If None, no logging is performed.
        Logs include:
        - Epoch progress (25%, 50%, 75%, 100% milestones)
        - Training and validation metrics after each epoch
        - Best model save notifications
    gpu_stft_params : dict, optional
        Parameters for GPU-accelerated spectrogram computation. If None, assumes
        data loader returns CPU-computed or pre-computed spectrograms. If provided, raw audio
        from dataloader will be converted to spectrograms on GPU during training.
        
        This is passed directly to `train_epoch()` - see that function's documentation
        for detailed parameter specifications.
        
        Required keys:
            'window' : torch.Tensor
                Pre-allocated window tensor on GPU, shape (n_window,). This is the applied window to the STFT.
            'hop_size' : int
                STFT hop size in samples.
            'mfft' : int
                FFT size computed using _compute_mfft(samples_per_frame, hop_size)
        
        Optional keys:
            'pos_encoding' : torch.Tensor or None
                Pre-computed positional encoding on GPU. See dataset.positional_encoding() for details.
    
    Returns
    -------
    dict
        Training history dictionary with the following keys:
        
        - 'train_loss' : np.ndarray of shape (num_epochs,)
            Average training loss per batch for each epoch
        - 'val_loss' : np.ndarray of shape (num_epochs,)
            Average validation loss for each epoch
        - 'val_position_mae' : np.ndarray of shape (num_epochs,)
            Mean absolute error of position predictions (meters) for each epoch
        - 'val_rotation_mae' : np.ndarray of shape (num_epochs,)
            Mean absolute error of quaternion predictions for each epoch
        - 'val_angular_error' : np.ndarray of shape (num_epochs,)
            Angular error in degrees for rotation predictions for each epoch
    
    Notes
    -----
    **Checkpointing**:
    Two checkpoint files are saved in `save_dir`:
    
    1. `{model_name}__best_model.pth`: Best model based on lowest validation loss
    2. `{model_name}__latest_trained_model.pth`: Final model from last epoch
    
    Each checkpoint contains:
    - 'epoch': Epoch number (0-indexed)
    - 'model_state_dict': Model parameters
    - 'optimizer_state_dict': Optimizer state
    - 'val_loss': Validation loss at that epoch
    - 'history': Complete training history
    - 'val_metrics': Validation metrics (best model only)
    
    **GPU Spectrogram Mode**:
    When `gpu_stft_params` is provided, spectrograms are computed on-the-fly on GPU
    during training, providing ~30x speedup over CPU scipy computation. This eliminates
    the data loading bottleneck and improves GPU utilization.
    
    **Metrics Tracked**:
    The function tracks multiple metrics for validation data:
    - Position MSE/MAE: Measures 3D position prediction accuracy
    - Rotation MSE/MAE: Measures quaternion prediction accuracy
    - Angular Error: Measures rotation error in degrees (more interpretable)
    
    **Progress Logging**:
    If a logger is provided, progress is logged at:
    - Start of each epoch
    - Training progress: 25%, 50%, 75%, 100% of batches
    - Validation progress: 33%, 66%, 100% of batches
    - End of epoch with all metrics
    - Best model save notifications
    
    **Model Requirements**:
    The model must implement:
    - `output_dim` attribute: Integer (3, 4, or 7) indicating prediction dimensionality
    - `getModelName()` method: Returns string name for checkpoint files
    
    See Also
    --------
    train_epoch : Single epoch training with detailed documentation
    evaluate : Model evaluation without gradient computation
    load_checkpoint : Load saved checkpoint to resume training
    apply_spectogram_batch_gpu : GPU spectrogram computation
    """
    os.makedirs(save_dir, exist_ok=True)
    # Check call to getModelName(), so it crashes fast before training:
    model.getModelName()

    best_val_loss = float("inf")
    history = {
        "train_loss": np.ones(num_epochs) * -1.0,
        "val_loss": np.ones(num_epochs) * -1.0,
        "val_position_mae": np.ones(num_epochs) * -1.0,
        "val_rotation_mae": np.ones(num_epochs) * -1.0,
        "val_angular_error": np.ones(num_epochs) * -1.0,
    }
    model.train()

    # Set up tqdm logging callback
    if logger:
        # Add tqdm-compatible handler temporarily
        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setFormatter(
            logger.handlers[0].formatter if logger.handlers else None
        )
        logger.addHandler(tqdm_handler)

    for epoch in range(num_epochs):
        epoch_header = f"\t\tEpoch {epoch+1}/{num_epochs}, on device '{device}'"
        if logger:
            logger.debug(epoch_header)

        train_loss = train_epoch(
            epoch=epoch,
            model=model,
            dataloader=train_loader,
            lossFunction=lossFunction,
            optimizer=optimizer,
            device=device,
            logger=logger,
            gpu_stft_params=gpu_stft_params,
        )

        val_metrics = evaluate(
            model, val_loader, lossFunction, device, logger=logger, gpu_stft_params=gpu_stft_params
        )

        history["train_loss"][epoch] = train_loss
        history["val_loss"][epoch] = val_metrics["loss"]
        history["val_position_mae"][epoch] = val_metrics["position_mae"]
        history["val_rotation_mae"][epoch] = val_metrics["rotation_mae"]
        history["val_angular_error"][epoch] = val_metrics["angular_error_deg"]

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

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = os.path.join(
                save_dir, f"{model.getModelName()}__best_model.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "history": history,
                },
                checkpoint_path,
            )

            best_model_msg = f"    Saved best model (val loss: {best_val_loss:.6f})"
            if logger:
                logger.debug(best_model_msg)

    # Save last model:
    checkpoint_best_path = os.path.join(
        save_dir, f"{model.getModelName()}__latest_trained_model.pth"
    )
    torch.save(
        {
            "epoch": epoch,  # current epoch
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "history": history,
        },
        checkpoint_best_path,
    )

    # Remove tqdm handler
    if logger:
        logger.removeHandler(tqdm_handler)

    return history


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    lossFunction: Callable,
    device: torch.device,
    evalTest=False,
    logger: Optional[logging.Logger] = None,
    gpu_stft_params: Optional[dict] = None,
) -> Dict[str, float]:
    """
    Evaluate model on validation or test data without gradient computation.
    
    Computes loss and various metrics (position/rotation MAE, angular error) for model
    evaluation. Supports both CPU-computed spectrograms and GPU on-the-fly computation.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate. Must have `output_dim` attribute (3, 4, or 7)
    dataloader : DataLoader
        Data loader providing (audio, pose) batches for evaluation
    lossFunction : Callable
        Loss function with signature: loss_fn(pred_pose, true_pose, output_dim) -> Tensor
    device : torch.device
        Target device for computation (e.g., 'cuda:0', 'cpu')
    evalTest : bool, default=False
        If True, displays "Testing" in progress bar, otherwise "Evaluating"
    logger : logging.Logger, optional
        Logger for progress tracking at 33%, 66%, 100% milestones
    gpu_stft_params : dict, optional
        Parameters for GPU-accelerated spectrogram computation. If None, assumes
        dataloader returns pre-computed spectrograms. Required keys: 'window', 
        'hop_size', 'mfft', 'window_sum'. Optional: 'pos_encoding'
    
    Returns
    -------
    dict
        Dictionary with averaged metrics:
        - 'loss': Average loss across all batches
        - 'position_mse': Mean squared error for position predictions
        - 'position_mae': Mean absolute error for position predictions (meters)
        - 'rotation_mse': Mean squared error for quaternion predictions
        - 'rotation_mae': Mean absolute error for quaternion predictions
        - 'angular_error_deg': Angular error in degrees for rotation predictions
    """
    model.eval()

    total_loss = 0.0
    output_dim = model.output_dim
    all_metrics = {
        "position_mse": np.full(len(dataloader), np.nan, dtype=np.float64),
        "position_mae": np.full(len(dataloader), np.nan, dtype=np.float64),
        "rotation_mse": np.full(len(dataloader), np.nan, dtype=np.float64),
        "rotation_mae": np.full(len(dataloader), np.nan, dtype=np.float64),
        "angular_error_deg": np.full(len(dataloader), np.nan, dtype=np.float64),
    }
    if evalTest:
        desc_ = "Testing" + " on GPU" if gpu_stft_params is not None else ""
    else:
        desc_ = "Evaluating" + " on GPU" if gpu_stft_params is not None else ""
    # Set up progress logging milestones
    total_batches = len(dataloader)
    milestones = {int(total_batches * 0.33), int(total_batches * 0.66), total_batches}
    logged_milestones = set()

    with torch.no_grad():
        pbar = tqdm.tqdm(
            dataloader,
            desc=desc_,
            leave=False,
            disable=not is_interactive_environment(),
        )
        for idx, (audio, pose) in enumerate(pbar):
            audio = audio.to(device)
            pose = pose.to(device)
            
            # Compute spectrograms on GPU if enabled
            if gpu_stft_params is not None:
                audio = apply_spectogram_batch_gpu(
                    audio,
                    window=gpu_stft_params['window'],
                    hop_size=gpu_stft_params['hop_size'],
                    mfft=gpu_stft_params['mfft'],
                    pos_encoding_tensor=gpu_stft_params.get('pos_encoding', None)
                )
                # Apply window normalization and 100x scaling in one operation (optimization)
                # scipy's ShortTimeFFT divides by window sum, then we multiply by 100
                audio = audio * (100.0 / gpu_stft_params['window_sum'])

            pred_pose = model(audio)
            loss = lossFunction(pred_pose, pose, output_dim)

            pos_loss = metrics_module.position_loss(pred_pose, pose, output_dim)
            rot_loss = metrics_module.rotation_loss(pred_pose, pose, output_dim)

            if rot_loss.item() > 0.0:
                ratio_text = f"{pos_loss.item()/rot_loss.item():.4f}"
            else:
                ratio_text = "N/A"

            pbar.set_postfix(
                {
                    "Pos Loss": f"{pos_loss.item():.4f}",
                    "Rot Loss": f"{rot_loss.item():.4f}",
                    "Ratio POS/ROT": ratio_text,
                }
            )
            total_loss += loss.item()
            metrics = metrics_module.compute_metrics(pred_pose, pose, output_dim)
            for key, value in metrics.items():
                all_metrics[key][idx] = value

            # Log progress at milestones
            if logger:
                for milestone in milestones:
                    if idx >= milestone and milestone not in logged_milestones:
                        percentage = (milestone / total_batches) * 100
                        logger.debug(
                            f"{desc_}: {percentage:.0f}% complete ({milestone}/{total_batches} batches)"
                        )
                        logged_milestones.add(milestone)

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics["loss"] = avg_loss

    return avg_metrics


def evaluate_per_samples(
    model: torch.nn.Module,
    dataloader: DataLoader,
    lossFunction: Callable,
    device: torch.device,
    evalTest=False,
    logger: Optional[logging.Logger] = None,
    gpu_stft_params: Optional[dict] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Evaluate model per sample, returning both averaged and per-sample metrics.
    
    Similar to evaluate() but computes metrics for each individual sample instead of
    averaging over batches. Useful for analyzing model performance distribution and
    identifying challenging samples. Supports GPU-accelerated spectrogram computation.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate. Must have `output_dim` attribute (3, 4, or 7)
    dataloader : DataLoader
        Data loader providing (audio, pose) batches for evaluation
    lossFunction : Callable
        Loss function with signature: loss_fn(pred_pose, true_pose, output_dim) -> Tensor
    device : torch.device
        Target device for computation (e.g., 'cuda:0', 'cpu')
    evalTest : bool, default=False
        If True, displays "Testing [per sample]" in progress bar, otherwise "Evaluating [per sample]"
    logger : logging.Logger, optional
        Logger for progress tracking at 25%, 50%, 75%, 100% milestones
    gpu_stft_params : dict, optional
        Parameters for GPU-accelerated spectrogram computation. If None, assumes
        dataloader returns pre-computed spectrograms. Required keys: 'window', 
        'hop_size', 'mfft', 'window_sum'. Optional: 'pos_encoding'
    
    Returns
    -------
    tuple[dict, dict]
        Two dictionaries:
        
        1. Averaged metrics (same as evaluate()):
           - 'loss': Average loss across all batches
           - 'position_mse': Mean squared error for position predictions
           - 'position_mae': Mean absolute error for position predictions (meters)
           - 'rotation_mse': Mean squared error for quaternion predictions
           - 'rotation_mae': Mean absolute error for quaternion predictions
           - 'angular_error_deg': Angular error in degrees for rotation predictions
        
        2. Per-sample metrics (all arrays of shape (num_samples,)):
           - 'position_mse': Per-sample position MSE
           - 'position_mae': Per-sample position MAE (meters)
           - 'rotation_mse': Per-sample rotation MSE
           - 'rotation_mae': Per-sample rotation MAE
           - 'angular_error_deg': Per-sample angular error (degrees)
    """
    model.eval()

    total_loss = 0.0
    output_dim = model.output_dim
    num_samples = len(dataloader.dataset)  # total samples, not batches

    all_metrics = {
        "position_mse": np.full(num_samples, np.nan, dtype=np.float64),
        "position_mae": np.full(num_samples, np.nan, dtype=np.float64),
        "rotation_mse": np.full(num_samples, np.nan, dtype=np.float64),
        "rotation_mae": np.full(num_samples, np.nan, dtype=np.float64),
        "angular_error_deg": np.full(num_samples, np.nan, dtype=np.float64),
    }
    if evalTest:
        desc_ = "Testing [per sample]"
    else:
        desc_ = "Evaluating [per sample]"

    # Set up progress logging milestones
    total_batches = len(dataloader)
    milestones = {
        int(total_batches * 0.25),
        int(total_batches * 0.5),
        int(total_batches * 0.75),
        total_batches,
    }
    logged_milestones = set()

    start_idx = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(
            dataloader,
            desc=desc_,
            leave=False,
            disable=not is_interactive_environment(),
        )
        for batch_idx, (audio_batch, pose_batch) in enumerate(pbar, 1):
            batch_size = audio_batch.size(0)
            end_idx = start_idx + batch_size

            audio = audio_batch.to(device)
            pose = pose_batch.to(device)
            
            # Compute spectrograms on GPU if enabled
            if gpu_stft_params is not None:
                audio = apply_spectogram_batch_gpu(
                    audio,
                    window=gpu_stft_params['window'],
                    hop_size=gpu_stft_params['hop_size'],
                    mfft=gpu_stft_params['mfft'],
                    pos_encoding_tensor=gpu_stft_params.get('pos_encoding', None)
                )
                # Apply window normalization and 100x scaling in one operation (optimization)
                # scipy's ShortTimeFFT divides by window sum, then we multiply by 100
                audio = audio * (100.0 / gpu_stft_params['window_sum'])

            pred_pose = model(audio)
            loss = lossFunction(pred_pose, pose, output_dim)

            pos_loss = metrics_module.position_loss(pred_pose, pose, output_dim)
            rot_loss = metrics_module.rotation_loss(pred_pose, pose, output_dim)

            if rot_loss.item() > 0.0:
                ratio_text = f"{pos_loss.item()/rot_loss.item():.4f}"
            else:
                ratio_text = "N/A"

            pbar.set_postfix(
                {
                    "Pos Loss": f"{pos_loss.item():.4f}",
                    "Rot Loss": f"{rot_loss.item():.4f}",
                    "Ratio POS/ROT": ratio_text,
                }
            )
            total_loss += loss.item()
            per_sample_metrics = metrics_module.compute_metrics_per_sample(
                pred_pose, pose, output_dim
            )
            for key, values in per_sample_metrics.items():
                all_metrics[key][start_idx:end_idx] = values
            start_idx = end_idx

            # Log progress at milestones
            if logger:
                for milestone in milestones:
                    if batch_idx >= milestone and milestone not in logged_milestones:
                        percentage = (milestone / total_batches) * 100
                        logger.debug(
                            f"{desc_}: {percentage:.0f}% complete ({milestone}/{total_batches} batches)"
                        )
                        logged_milestones.add(milestone)

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics["loss"] = avg_loss

    return avg_metrics, all_metrics


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch_checkpoint = checkpoint["epoch"]
    # history_checkpoint = checkpoint['history']

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f" Loaded checkpoint from epoch {epoch_checkpoint + 1}")
    print(f" Validation loss: {checkpoint['val_loss']:.6f}")

    return checkpoint
