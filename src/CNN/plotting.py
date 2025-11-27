from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import logging
from src.CNN.cnn_model import CNNModelParams


def plot_training_history(history: Dict[str, List], save_path: str = 'training_results.png',
                          logger: Optional[logging.Logger] = None):
    """
    Plot training history with logging support.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger instance for logging plot save information
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Position MAE
    axes[0, 1].plot(history['val_position_mae'], label='Val Position MAE',
                    color='orange', marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Position MAE (meters)')
    axes[0, 1].set_title('Validation Position MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Rotation MAE
    axes[1, 0].plot(history['val_rotation_mae'], label='Val Quaternion MAE',
                    color='green', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Quaternion MAE')
    axes[1, 0].set_title('Validation Rotation MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Angular Error
    axes[1, 1].plot(history['val_angular_error'], label='Val Angular Error',
                    color='red', marker='o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Angular Error (degrees)')
    axes[1, 1].set_title('Validation Angular Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_test_results(
    avg_metrics: Dict[str, float],
    history_per_sample: Dict[str, np.ndarray],
    model_params: Optional[CNNModelParams] = None,
    save_path: str = 'test_results.png',
    logger: Optional[logging.Logger] = None
):
    """
    Plot test evaluation results with average metrics and per-sample history.

    Creates a 2x2 grid with:
        Row 1, Left: Average test metrics display (text)
        Row 1, Right: Model parameters display (text)
        Row 2, Left: Position MSE per sample (line plot)
        Row 2, Right: Angular error per sample (line plot)

    Parameters
    ----------
    avg_metrics : Dict[str, float]
        Dictionary of average metrics including:
        - loss
        - position_mse
        - position_mae
        - rotation_mse
        - rotation_mae
        - angular_error_deg

    history_per_sample : Dict[str, np.ndarray]
        Dictionary of per-sample metric arrays including:
        - position_mse
        - position_mae
        - rotation_mse
        - rotation_mae
        - angular_error_deg

    model_params : CNNModelParams, optional
        Model parameters to display.

    save_path : str
        Path where the plot will be saved (default: 'test_results.png').

    logger : logging.Logger, optional
        Logger instance for logging save information

    Example
    -------
    >>> avg_metrics, history = train.evaluate_per_samples(model, test_loader, loss_fn, device)
    >>> plot_test_results(avg_metrics, history, model_params, 'test_results.png')
    """
    fig = plt.figure(figsize=(16, 10))

    # Create grid: 2 rows x 2 columns
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ========================================================================
    # Row 1, Left: Average Test Metrics (Text Display)
    # ========================================================================
    ax_metrics = fig.add_subplot(gs[0, 0])
    ax_metrics.axis('off')

    metrics_text = "Average Test Metrics\n" + "="*40 + "\n\n"

    if 'loss' in avg_metrics:
        metrics_text += f"Loss: {avg_metrics['loss']:.6f}\n\n"

    metrics_text += "Position Metrics:\n"
    if 'position_mse' in avg_metrics:
        metrics_text += f"  MSE: {avg_metrics['position_mse']:.6f} [m]\n"
    if 'position_mae' in avg_metrics:
        metrics_text += f"  MAE: {avg_metrics['position_mae']:.6f} [m]\n"

    metrics_text += "\nRotation Metrics:\n"
    if 'rotation_mse' in avg_metrics:
        metrics_text += f"  Quaternion MSE: {avg_metrics['rotation_mse']:.6f}\n"
    if 'rotation_mae' in avg_metrics:
        metrics_text += f"  Quaternion MAE: {avg_metrics['rotation_mae']:.6f}\n"
    if 'angular_error_deg' in avg_metrics:
        metrics_text += f"  Angular Error: {avg_metrics['angular_error_deg']:.2f}°\n"

    # Get number of samples
    num_samples = len(next(iter(history_per_sample.values())))
    metrics_text += f"\nTotal Samples: {num_samples}"

    ax_metrics.text(0.05, 0.95, metrics_text,
                   transform=ax_metrics.transAxes,
                   fontsize=11,
                   verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ========================================================================
    # Row 1, Right: Model Parameters (Text Display)
    # ========================================================================
    ax_params = fig.add_subplot(gs[0, 1])
    ax_params.axis('off')

    if model_params is not None:
        params_text = "Model Parameters\n" + "="*40 + "\n\n"
        params_text += f"Model Name: {model_params.model_name}\n"
        params_text += f"Input Channels: {model_params.n_channels}\n"
        params_text += f"Samples/Frame: {model_params.samples_per_frame}\n\n"

        params_text += "CNN Architecture:\n"
        params_text += f"  Filters: {model_params.cnn_num_filter_list}\n"
        params_text += f"  Kernel Sizes: {model_params.cnn_filter_size_list}\n"
        params_text += f"  Strides: {model_params.cnn_stride_list}\n"
        params_text += f"  Padding: {model_params.cnn_padding_list}\n"
        params_text += f"  MaxPool K: {model_params.max_pool_filter_size_list}\n"
        params_text += f"  MaxPool S: {model_params.max_pool_stride_size_list}\n\n"

        params_text += "FC Architecture:\n"
        params_text += f"  Hidden Dims: {model_params.FC_hidden_dims}\n"
        params_text += f"  Output Dim: {model_params.output_dim}\n"
        params_text += f"  Dropout: {model_params.dropout}"

        ax_params.text(0.05, 0.95, params_text,
                      transform=ax_params.transAxes,
                      fontsize=10,
                      verticalalignment='top',
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    else:
        ax_params.text(0.5, 0.5, "No Model Parameters\nProvided",
                      transform=ax_params.transAxes,
                      fontsize=12,
                      ha='center',
                      va='center',
                      style='italic')

    # ========================================================================
    # Row 2, Left: Position MSE per Sample
    # ========================================================================
    ax_pos = fig.add_subplot(gs[1, 0])

    if 'position_mse' in history_per_sample:
        sample_indices = np.arange(len(history_per_sample['position_mse']))
        position_mse = history_per_sample['position_mse']
        num_samples = len(position_mse)

        ax_pos.plot(sample_indices, position_mse,
                   color='blue', linewidth=0.8, alpha=0.7)

        # Add average line
        avg_pos_mse = avg_metrics.get('position_mse', np.mean(position_mse))
        ax_pos.axhline(y=avg_pos_mse, color='red', linestyle='--',
                      linewidth=2, label=f'Average Euclidian Error: {avg_pos_mse:.6f} [m]')

        ax_pos.set_xlabel('Sample Number', fontsize=11)
        ax_pos.set_ylabel('Euclidian [m]', fontsize=11)
        ax_pos.set_title('Euclidian/Position "MSE" per Sample', fontsize=12, fontweight='bold')
        ax_pos.set_xlim(0, num_samples)
        ax_pos.legend(loc='upper right')
        ax_pos.grid(True, alpha=0.3)
    else:
        ax_pos.text(0.5, 0.5, "No Position MSE\nData Available",
                   transform=ax_pos.transAxes,
                   fontsize=12, ha='center', va='center', style='italic')

    # ========================================================================
    # Row 2, Right: Angular Error per Sample
    # ========================================================================
    ax_ang = fig.add_subplot(gs[1, 1])

    if 'angular_error_deg' in history_per_sample:
        sample_indices = np.arange(len(history_per_sample['angular_error_deg']))
        angular_error = history_per_sample['angular_error_deg']
        num_samples = len(angular_error)

        ax_ang.plot(sample_indices, angular_error,
                   color='red', linewidth=0.8, alpha=0.7)

        # Add average line
        avg_ang_error = avg_metrics.get('angular_error_deg', np.mean(angular_error))
        ax_ang.axhline(y=avg_ang_error, color='green', linestyle='--',
                      linewidth=2, label=f'Average Angular error: {avg_ang_error:.2f}°')

        ax_ang.set_xlabel('Sample Number', fontsize=11)
        ax_ang.set_ylabel('Angular Error (degrees)', fontsize=11)
        ax_ang.set_title('Angular Error per Sample', fontsize=12, fontweight='bold')
        ax_ang.set_xlim(0, num_samples)
        ax_ang.legend(loc='upper right')
        ax_ang.grid(True, alpha=0.3)
    else:
        ax_ang.text(0.5, 0.5, "No Angular Error\nData Available",
                   transform=ax_ang.transAxes,
                   fontsize=12, ha='center', va='center', style='italic')

    # Add main title
    fig.suptitle('Test Results Analysis', fontsize=14, fontweight='bold', y=0.98)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    save_msg = f"      Test results plot saved to: {save_path}"
    if logger:
        logger.debug(save_msg)

    plt.close()


def print_metrics(metrics: Dict[str, float], title: str = "Metrics", logger: Optional[logging.Logger] = None):
    """
    Print metrics with logging support.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger instance for logging metrics
    """
    header = f"\n\t{'='*60}\n\t{title}\n\t{'='*60}"
    metrics_output = []

    if 'loss' in metrics:
        metrics_output.append(f"\tLoss: {metrics['loss']:.6f}")

    metrics_output.append("\n\tPosition Metrics:")
    if 'position_mse' in metrics:
        metrics_output.append(f"\t  MSE: {metrics['position_mse']:.6f}")
    if 'position_mae' in metrics:
        metrics_output.append(f"\t  MAE: {metrics['position_mae']:.6f} meters")

    metrics_output.append("\n\tRotation Metrics:")
    if 'rotation_mse' in metrics:
        metrics_output.append(f"\t  Quaternion MSE: {metrics['rotation_mse']:.6f}")
    if 'rotation_mae' in metrics:
        metrics_output.append(f"\t  Quaternion MAE: {metrics['rotation_mae']:.6f}")
    if 'angular_error_deg' in metrics:
        metrics_output.append(f"\t  Angular Error: {metrics['angular_error_deg']:.2f}°")

    footer = "\t" + '='*60

    full_message = f"{header}\n" + "\n".join(metrics_output) + f"\n{footer}"

    if logger:
        logger.debug(full_message)
