# Created Nov 8th, 2025
# Author: Jaduk Suh
import torch, torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import json
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Try to import wandb, but make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for experiment tracking.")

# Import model components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.feature_extractors import LinearExtractor
from models.sequence import LSTMSeq
from models.heads import LinearHead

# Import utility functions
from utils.utilsIO import (
    get_paths, get_head_tracking_fs, get_corresponding_array_orientation_data,
    get_unique_part_IDs, unpack_6DOF_data, convert_int_to_float
)

# Constants
SAMPLE_RATE = 48000
FRAME_LEN = 0.05
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_LEN)  # 2400


class AudioPoseDataset(Dataset):
    """Dataset for audio-pose pairs from EasyCom dataset."""
    
    def __init__(self, audio_dir, pose_dir, session_ids, participant_id=2):
        """
        Args:
            audio_dir: Path to Glasses_Microphone_Array_Audio directory
            pose_dir: Path to Tracked_Poses directory
            session_ids: List of session IDs to include (e.g., [1, 2, ..., 10])
            participant_id: Participant ID to extract pose for (default: 2)
        """
        self.audio_dir = Path(audio_dir)
        self.pose_dir = Path(pose_dir)
        self.session_ids = session_ids
        self.participant_id = participant_id
        
        self.fs_head_tracking = get_head_tracking_fs()
        self.dT_head_tracking = 1.0 / self.fs_head_tracking
        
        # Collect all audio-pose pairs
        self.samples = []
        for session_id in session_ids:
            session_name = f"Session_{session_id}"
            session_audio_dir = self.audio_dir / session_name
            session_pose_dir = self.pose_dir / session_name
            
            if not session_audio_dir.exists() or not session_pose_dir.exists():
                print(f"Warning: {session_name} not found, skipping...")
                continue
                
            for wav_file in session_audio_dir.glob("*.wav"):
                pose_file = session_pose_dir / (wav_file.stem + ".json")
                if pose_file.exists():
                    self.samples.append((wav_file, pose_file, session_name))
        
        print(f"Loaded {len(self.samples)} audio-pose pairs from sessions {session_ids}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        wav_file, pose_file, session_name = self.samples[idx]
        
        # Read audio
        audio_data, sr = torchaudio.load(wav_file)
        
        # Handle mono vs multi-channel audio
        if len(audio_data.shape) == 1:
            audio_data = audio_data[:, np.newaxis]
        
        N_channels, N_taps = audio_data.shape
        t_max = N_taps / sr
        N_frames = int(round(t_max / self.dT_head_tracking))
        N_samples_per_frame = int(sr * self.dT_head_tracking)
        
        # Reshape audio to (frames, channels, samples_per_frame)
        audio_tensor = torch.stack([
            audio_data[:, i * N_samples_per_frame:(i+1)*N_samples_per_frame] for i in range(N_frames)
        ], dim=0)
        
        # Read pose data
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)

        pose_tensor = torch.zeros((len(pose_data), 7))
        for i, frame in enumerate(pose_data):
            assert i == frame['Frame_Number'] - 1
            for participant in frame['Participants']:
                if participant['Participant_ID'] != 2:
                    continue
                else:
                    pose_tensor[i, 0] = participant["Position_X"]
                    pose_tensor[i, 1] = participant["Position_Y"]
                    pose_tensor[i, 2] = participant["Position_Z"]
                    pose_tensor[i, 3] = participant["Quaternion_X"]
                    pose_tensor[i, 4] = participant["Quaternion_Y"]
                    pose_tensor[i, 5] = participant["Quaternion_Z"]
                    pose_tensor[i, 6] = participant["Quaternion_W"]
        
        # Ensure frames match
        min_frames = min(audio_tensor.shape[0], pose_tensor.shape[0])
        audio_tensor = audio_tensor[:min_frames, :]
        pose_tensor = pose_tensor[:min_frames, :]
       
        return audio_tensor, pose_tensor


class AudioPoseModel(nn.Module):
    """Full model combining LinearExtractor, LSTMSeq, and LinearHead."""
    
    def __init__(self, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_channels = 6
        self.feature_extractor = LinearExtractor(input_dim=2400, hidden_dim=hidden_dim)
        self.sequence_model = LSTMSeq(hidden_dim=hidden_dim * self.num_channels, num_layers=num_layers, dropout=dropout)
        self.head = LinearHead(hidden_dim=hidden_dim * self.num_channels)
    
    def forward(self, x):
        """
        Args:
            x: Audio tensor of shape (batch, seq_len, samples_per_frame)
        Returns:
            Output tensor of shape (batch, seq_len, 7) - [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        """
        batch_size, seq_len, num_channels, samples_per_frame = x.shape
        assert num_channels == self.num_channels
        
        # Process each frame through feature extractor
        # Reshape to (batch * seq_len * num_channels, samples_per_frame)
        x_flat = x.view(-1, samples_per_frame)
        # Extract features: (batch * seq_len * num_channels, hidden_dim)
        features = self.feature_extractor(x_flat)
        # Reshape back to (batch, seq_len, num_channels * hidden_dim)
        features = features.view(batch_size, seq_len, -1)
        
        # Process sequence through LSTM
        # Output: (batch, seq_len, num_channels * hidden_dim)
        seq_features = self.sequence_model(features)
        
        # Apply head to each timestep
        # Reshape to (batch * seq_len, num_channels * hidden_dim)
        seq_features_flat = torch.reshape(seq_features, (-1, seq_features.shape[-1]))
        # Output: (batch * seq_len, 7)
        output = self.head(seq_features_flat)
        # Reshape back to (batch, seq_len, 7)
        output = output.view(batch_size, seq_len, -1)
        
        return output


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences.
    
    Args:
        batch: List of tuples (audio, pose) where:
            - audio: tensor of shape [seq_len, num_channel, input_dim]
            - pose: tensor of shape [seq_len, 7]
    
    Returns:
        audio_batch: Padded tensor of shape [batch_size, max_seq_len, num_channel, input_dim]
        pose_batch: Padded tensor of shape [batch_size, max_seq_len, 7]
    """
    audio_list, pose_list = zip(*batch)
    
    # Pad sequences to the maximum length in the batch
    # pad_sequence expects (seq_len, *) tensors and pads along the first dimension
    audio_batch = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
    pose_batch = pad_sequence(pose_list, batch_first=True, padding_value=0.0)
    
    return audio_batch, pose_batch


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for audio, pose in dataloader:
        audio = audio.to(device)
        pose = pose.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(audio)
        
        # Compute loss (only on non-padded frames)
        # Create mask for valid frames (frames where pose is not all zeros)
        # Sum across pose dimensions, then check if > 0
        valid_mask = (pose.abs().sum(dim=-1) > 1e-6)  # (batch, seq_len)
        if valid_mask.sum() > 0:
            # Only compute loss on valid frames
            loss = criterion(output[valid_mask], pose[valid_mask])
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set.
    
    Returns:
        dict: Dictionary containing:
            - 'loss': Mean squared error loss
            - 'positional_error': Mean Euclidean distance error for positions (in meters)
            - 'angular_error': Mean angular error for orientations (in degrees)
    """
    model.eval()
    total_loss = 0.0
    total_positional_error = 0.0
    total_angular_error = 0.0
    total_valid_frames = 0
    num_batches = 0
    
    with torch.no_grad():
        for audio, pose in dataloader:
            audio = audio.to(device)
            pose = pose.to(device)
            
            # Forward pass
            output = model(audio)
            
            # Compute loss (only on non-padded frames)
            valid_mask = (pose.abs().sum(dim=-1) > 1e-6)  # (batch, seq_len)
            if valid_mask.sum() > 0:
                # Extract valid predictions and ground truth
                valid_output = output[valid_mask]  # (N_valid, 7)
                valid_pose = pose[valid_mask]      # (N_valid, 7)
                
                # Compute MSE loss
                loss = criterion(valid_output, valid_pose)
                total_loss += loss.item()
                
                # Compute positional error (Euclidean distance for first 3 dimensions)
                pred_pos = valid_output[:, :3]  # (N_valid, 3)
                gt_pos = valid_pose[:, :3]      # (N_valid, 3)
                positional_errors = torch.norm(pred_pos - gt_pos, dim=1)  # (N_valid,)
                total_positional_error += positional_errors.sum().item()
                
                # Compute angular error (angle between quaternions)
                pred_quat = valid_output[:, 3:]  # (N_valid, 4) [x, y, z, w]
                gt_quat = valid_pose[:, 3:]      # (N_valid, 4) [x, y, z, w]
                
                # Normalize quaternions
                pred_quat = pred_quat / (torch.norm(pred_quat, dim=1, keepdim=True) + 1e-8)
                gt_quat = gt_quat / (torch.norm(gt_quat, dim=1, keepdim=True) + 1e-8)
                
                # Compute dot product (clamp to [-1, 1] for numerical stability)
                dot_product = torch.clamp(torch.sum(pred_quat * gt_quat, dim=1), -1.0, 1.0)
                
                # Angular error in radians (using 2 * arccos(|dot|) for quaternion distance)
                # We use absolute value to handle quaternion double-cover (q and -q represent same rotation)
                angular_errors_rad = 2 * torch.acos(torch.abs(dot_product))
                
                # Convert to degrees
                angular_errors_deg = torch.rad2deg(angular_errors_rad)
                total_angular_error += angular_errors_deg.sum().item()
                
                total_valid_frames += valid_mask.sum().item()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=False)
                total_loss += loss.item()
            
            num_batches += 1
    
    # Compute averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_positional_error = total_positional_error / total_valid_frames if total_valid_frames > 0 else 0.0
    avg_angular_error = total_angular_error / total_valid_frames if total_valid_frames > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'positional_error': avg_positional_error,
        'angular_error': avg_angular_error
    }


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


def main():
    # Configuration
    data_root = Path("data/Main")
    audio_dir = data_root / "Glasses_Microphone_Array_Audio"
    pose_dir = data_root / "Tracked_Poses"
    
    # Train/Dev/Test splits
    train_sessions = list(range(1, 11))  # Sessions 1-10
    dev_sessions = [11]                  # Session 11
    test_sessions = [12]                  # Session 12
    
    # Model hyperparameters
    hidden_dim = 32
    num_layers = 2
    dropout = 0.1
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-3
    
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = AudioPoseDataset(audio_dir, pose_dir, train_sessions, participant_id=2)
    dev_dataset = AudioPoseDataset(audio_dir, pose_dir, dev_sessions, participant_id=2)
    test_dataset = AudioPoseDataset(audio_dir, pose_dir, test_sessions, participant_id=2)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Set device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Running on device {device}")

    # Create model
    model = AudioPoseModel(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize wandb if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="audio-pose-prediction",
            config={
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "train_sessions": train_sessions,
                "dev_sessions": dev_sessions,
                "test_sessions": test_sessions,
            }
        )
        wandb.watch(model)
    
    # Track training history
    train_losses = []
    dev_losses = []
    dev_positional_errors = []
    dev_angular_errors = []
    
    # Training loop
    print("Starting training...")
    best_dev_loss = float('inf')
    
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_metrics = evaluate(model, dev_loader, criterion, device)
        
        # Track history
        train_losses.append(train_loss)
        dev_losses.append(dev_metrics['loss'])
        dev_positional_errors.append(dev_metrics['positional_error'])
        dev_angular_errors.append(dev_metrics['angular_error'])
        
        # Log to wandb if available
        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "dev_loss": dev_metrics['loss'],
                "dev_positional_error": dev_metrics['positional_error'],
                "dev_angular_error": dev_metrics['angular_error'],
            })
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Dev Loss: {dev_metrics['loss']:.6f}")
        print(f"  Dev Positional Error: {dev_metrics['positional_error']:.4f} m")
        print(f"  Dev Angular Error: {dev_metrics['angular_error']:.4f}°")
        
        # Save best model
        if dev_metrics['loss'] < best_dev_loss:
            best_dev_loss = dev_metrics['loss']
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  Saved best model (dev loss: {dev_metrics['loss']:.6f})")
        print()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    model.load_state_dict(torch.load("best_model.pth"))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.6f}")
    print(f"Test Positional Error: {test_metrics['positional_error']:.4f} m")
    print(f"Test Angular Error: {test_metrics['angular_error']:.4f}°")
    
    # Log test metrics to wandb
    if WANDB_AVAILABLE:
        wandb.log({
            "test_loss": test_metrics['loss'],
            "test_positional_error": test_metrics['positional_error'],
            "test_angular_error": test_metrics['angular_error'],
        })
        wandb.finish()
    
    # Save training history and plot curves
    save_training_history(train_losses, dev_losses, dev_positional_errors, dev_angular_errors, test_metrics)
    plot_training_curves(train_losses, dev_losses)


if __name__ == '__main__':
    main()
