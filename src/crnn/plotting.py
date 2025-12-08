from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history: Dict[str, List], save_path: str = 'training_results.png'):
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


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    print(f"\n{'='*60}")
    print(title)
    print('='*60)
    
    if 'loss' in metrics:
        print(f"Loss: {metrics['loss']:.6f}")
    
    print(f"\nPosition Metrics:")
    if 'position_mse' in metrics:
        print(f"  MSE: {metrics['position_mse']:.6f}")
    if 'position_mae' in metrics:
        print(f"  MAE: {metrics['position_mae']:.6f} meters")
    
    print(f"\nRotation Metrics:")
    if 'rotation_mse' in metrics:
        print(f"  Quaternion MSE: {metrics['rotation_mse']:.6f}")
    if 'rotation_mae' in metrics:
        print(f"  Quaternion MAE: {metrics['rotation_mae']:.6f}")
    if 'angular_error_deg' in metrics:
        print(f"  Angular Error: {metrics['angular_error_deg']:.2f}°")
    
    print('='*60)
