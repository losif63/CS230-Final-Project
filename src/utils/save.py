import matplotlib.pyplot as plt
import json
from datetime import datetime

def plot_training_curves(train_losses, dev_losses, save_path="training_curves.png"):
    """Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        dev_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    plt.plot(epochs, dev_losses, 'r-', label='Dev Loss', linewidth=2, marker='s')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def save_training_history(train_losses, dev_losses, dev_positional_errors, dev_angular_errors, 
                         test_metrics, save_path="training_history.json"):
    """Save training history to JSON file.
    
    Args:
        train_losses: List of training losses per epoch
        dev_losses: List of validation losses per epoch
        dev_positional_errors: List of validation positional errors per epoch
        dev_angular_errors: List of validation angular errors per epoch
        test_metrics: Dictionary with test set metrics
        save_path: Path to save the JSON file
    """
    history = {
        "train_losses": train_losses,
        "dev_losses": dev_losses,
        "dev_positional_errors": dev_positional_errors,
        "dev_angular_errors": dev_angular_errors,
        "test_loss": test_metrics['loss'],
        "test_positional_error": test_metrics['positional_error'],
        "test_angular_error": test_metrics['angular_error'],
        "num_epochs": len(train_losses),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved to {save_path}")