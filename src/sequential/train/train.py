# Created Nov 8th, 2025
# Author: Jaduk Suh
import math
import torch, torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from pathlib import Path
import json
import numpy as np
import random
from tqdm import tqdm
from typing import Dict
import argparse

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
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.feature_extractors import LinearExtractor, MLPExtractor
from models.sequence import LSTMSeq, TransformerSeq
from models.heads import LinearHead, MLPHead

# Import utility functions
from utils.utilsIO import get_head_tracking_fs
from utils.save import save_training_history, plot_training_curves

from baseline.metrics import pose_6dof_loss
from baseline.data_utils import EasyComDataLoader

# Constants
SAMPLE_RATE = 48000
FRAME_LEN = 0.05
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_LEN)  # 2400


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TrainConfig():
    def __init__(self, config: Dict):
        self.batch_size = config["batch_size"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.num_epochs = config["num_epochs"]
        self.learning_rate = config["learning_rate"]

        self.feature_extractor = config["feature_extractor"]
        self.sequence_model = config["sequence_model"]
        self.head = config["head"]
    
    def __str__(self):
        def classname(x):
            # Handles both classes and instances
            return x.__name__ if isinstance(x, type) else x.__class__.__name__

        return (
            "TrainConfig(\n"
            f"  batch_size={self.batch_size},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  dropout={self.dropout},\n"
            f"  num_epochs={self.num_epochs},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  feature_extractor={classname(self.feature_extractor)},\n"
            f"  sequence_model={classname(self.sequence_model)},\n"
            f"  head={classname(self.head)}\n"
            ")"
        )

    def to_dict(self):
        def classname(x):
            # Handles both classes and instances
            return x.__name__ if isinstance(x, type) else x.__class__.__name__ 
        """Convert to a JSON-serializable dict (classes → string names)."""
        return {
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "feature_extractor": classname(self.feature_extractor),
            "sequence_model": classname(self.sequence_model),
            "head": classname(self.head),
        }

class AudioPoseDataset(Dataset):
    def __init__(self, cache_dir, window_length):
        self.cache_dir = Path(cache_dir)
        self.window_length = window_length
        self.files = torch.load(self.cache_dir / "index.pt")
        self.audios = []
        self.poses = []
        self.valid_indices = []
        self.loader = EasyComDataLoader(data_root="data/Main")
        for file_idx, file in enumerate(tqdm(self.files)):
            path = self.cache_dir / file
            data = torch.load(path, map_location="cpu")
            assert data['audio'].shape[0] == data['pose'].shape[0]
            seq_len = data['audio'].shape[0]
            self.audios.append(data["audio"])
            self.poses.append(data['pose'])

            pose_file = self.loader.tracked_poses_dir / f"{file[:-3]}.json"
            with open(pose_file, 'r', encoding='latin-1') as f:
                poses_data = json.load(f)
            
            trans_file = self.loader.speech_transcriptions_dir / f"{file[:-3]}.json"
            with open(trans_file, 'r', encoding='latin-1') as f:
                transcription_data = json.load(f)
            speech_lookup = self.loader.create_speech_lookup(transcription_data, seq_len)
            participant_ids = self.loader.get_all_participant_ids(poses_data)

            for frame_idx in range(seq_len):
                is_active = any(
                    speech_lookup[pid][frame_idx]
                    for pid in participant_ids
                    if pid != self.loader.ARRAY_WEARER_ID
                )
                if is_active:
                    self.valid_indices.append((file_idx, frame_idx))
            
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        file_idx, frame_idx = self.valid_indices[idx]
        return self.audios[file_idx][max(0, frame_idx-self.window_length + 1):frame_idx+1], self.poses[file_idx][frame_idx]


class AudioPoseModel(nn.Module):
    """Full model combining LinearExtractor, LSTMSeq, and LinearHead."""
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.num_channels = 6
        self.feature_extractor = config.feature_extractor(input_dim=2400, hidden_dim=config.hidden_dim, num_channels=self.num_channels)
        self.sequence_model = config.sequence_model(hidden_dim=config.hidden_dim, num_layers=config.num_layers, dropout=config.dropout)
        self.head = config.head(hidden_dim=config.hidden_dim)
    
    def forward(self, x, lengths):
        """
        Args:
            x: Audio tensor of shape (batch, seq_len, samples_per_frame)
        Returns:
            Output tensor of shape (batch, seq_len, 7) - [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        """
        batch_size, seq_len, num_channels, samples_per_frame = x.shape
        assert num_channels == self.num_channels
        
        # Process each frame through feature extractor
        # Extract features: (batch, seq_len, hidden_dim)
        features = self.feature_extractor(x)
        
        # Process sequence through LSTM
        # Output: (batch, hidden_dim)
        features = self.sequence_model(features, lengths)
        
        # Apply head to each timestep
        # Output: (batch, 7)
        output = self.head(features)

        position = output[:, :3]
        quaternion = output[:, 3:]
        quaternion = torch.nn.functional.normalize(quaternion, p=2, dim=-1)
        output = torch.cat([position, quaternion], dim=-1)
        
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

    lengths = torch.tensor([a.shape[0] for a in audio_list], dtype=torch.long) 
    # Pad sequences to the maximum length in the batch
    # pad_sequence expects (seq_len, *) tensors and pads along the first dimension
    audio_batch = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
    pose_batch = pad_sequence(pose_list, batch_first=True, padding_value=0.0)
    
    return audio_batch, pose_batch, lengths

def train_step(model, audio, pose, lengths, criterion, optimizer, scheduler, device):
    """Train for one step."""
    model.train()
    
    # Move to device (non_blocking for faster transfer if using GPU)
    audio = audio.to(device, non_blocking=True)
    pose = pose.to(device, non_blocking=True)
    lengths = lengths.to(device, non_blocking=True)
    
    # Forward pass
    optimizer.zero_grad()
    output = model(audio, lengths)
    
    # Compute loss (only on non-padded and valid frames)
    # Create mask for valid frames:
    # 1. Not padded (pose is not all zeros)
    # 2. Pose norm >= 0.1 (filter out near-zero/invalid poses, like baseline)
    pose_norm = torch.norm(pose, dim=-1)  # (batch, seq_len)
    not_padded = (pose.abs().sum(dim=-1) > 1e-6)  # (batch, seq_len)
    is_valid_pose = (pose_norm >= 0.1)  # Filter low-norm poses like baseline
    valid_mask = not_padded & is_valid_pose  # (batch, seq_len)
    total_valid_frames = 0
    total_positional_error = 0.0
    total_angular_error = 0.0

    if valid_mask.sum() > 0:
        valid_output = output[valid_mask]  # (N_valid, 7)
        valid_pose = pose[valid_mask]      # (N_valid, 7)
        # Only compute loss on valid frames
        loss = criterion(valid_output, valid_pose)
        # Compute positional error (Euclidean distance for first 3 dimensions)
        pred_pos = valid_output[:, :3]  # (N_valid, 3)
        gt_pos = valid_pose[:, :3]      # (N_valid, 3)
        positional_errors = torch.norm(pred_pos - gt_pos, dim=1)  # (N_valid,)
        total_positional_error += positional_errors.sum().item()
        
        # Compute angular error (angle between quaternions)
        pred_quat = valid_output[:, 3:]  # (N_valid, 4) [x, y, z, w]
        gt_quat = valid_pose[:, 3:]      # (N_valid, 4) [x, y, z, w]
        
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
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    return {
        'loss': loss.item(),
        'positional_error': total_positional_error,
        'angular_error': total_angular_error,
        'valid_frames': total_valid_frames
    }



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
        for audio, pose, lengths in tqdm(dataloader):
            # Move to device (non_blocking for faster transfer if using GPU)
            audio = audio.to(device, non_blocking=True)
            pose = pose.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            
            # Forward pass
            output = model(audio, lengths)
            
            # Compute loss (only on non-padded and valid frames)
            # Create mask for valid frames:
            # 1. Not padded (pose is not all zeros)
            # 2. Pose norm >= 0.1 (filter out near-zero/invalid poses, like baseline)
            pose_norm = torch.norm(pose, dim=-1)  # (batch, seq_len)
            not_padded = (pose.abs().sum(dim=-1) > 1e-6)  # (batch, seq_len)
            is_valid_pose = (pose_norm >= 0.1)  # Filter low-norm poses like baseline
            valid_mask = not_padded & is_valid_pose  # (batch, seq_len)
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
                
                # Compute dot product (clamp to [-1, 1] for numerical stability)
                dot_product = torch.clamp(torch.sum(pred_quat * gt_quat, dim=1), -1.0, 1.0)
                
                # Angular error in radians (using 2 * arccos(|dot|) for quaternion distance)
                # We use absolute value to handle quaternion double-cover (q and -q represent same rotation)
                angular_errors_rad = 2 * torch.acos(torch.abs(dot_product))
                
                # Convert to degrees
                angular_errors_deg = torch.rad2deg(angular_errors_rad)
                total_angular_error += angular_errors_deg.sum().item()
                
                total_valid_frames += valid_mask.sum().item()
                
                # Explicitly delete intermediate tensors
                del valid_output, valid_pose, pred_pos, gt_pos, positional_errors
                del pred_quat, gt_quat, dot_product, angular_errors_rad, angular_errors_deg
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=False)
                total_loss += loss.item()
            
            # Explicitly delete batch tensors
            del audio, pose, lengths, output, loss, valid_mask
            
            num_batches += 1
    
    # Clear CUDA cache after evaluation
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Compute averages
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_positional_error = total_positional_error / total_valid_frames if total_valid_frames > 0 else 0.0
    avg_angular_error = total_angular_error / total_valid_frames if total_valid_frames > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'positional_error': avg_positional_error,
        'angular_error': avg_angular_error
    }

def run_experiment(config: TrainConfig,
                   train_dataset,
                   dev_dataset,
                   test_dataset,
                   run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config for bookkeeping
    with open(run_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Dataloaders
    # Use num_workers=0 since dataset is already fully loaded in memory
    # This avoids memory duplication from worker processes
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False)
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False)

    # Device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    print(f"[{run_dir.name}] Running on device {device}")

    # Model / optimizer / loss
    model = AudioPoseModel(config=config).to(device)
    criterion = pose_6dof_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps=(int)(0.025 * total_steps)

    warmup = LinearLR(optimizer=optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer=optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer=optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # wandb (optional)
    if WANDB_AVAILABLE:
        wandb.init(
            project="audio-pose-prediction",
            name=f"lr_{config.learning_rate:.0e}",
            config=config.to_dict(),
        )
        wandb.watch(model)

    train_losses = []
    train_positional_errors = []
    train_angular_errors = []
    dev_losses = []
    dev_positional_errors = []
    dev_angular_errors = []

    best_dev_loss = float('inf')
    best_model_path = run_dir / "best_model.pth"

    print(f"[{run_dir.name}] Starting training with config:\n{config}")

    global_step = 0
    eval_per_n_steps = 300
    total_train_loss = 0.0
    total_train_positional_error = 0.0
    total_train_angular_error = 0.0
    train_frames = 0
    for epoch in tqdm(range(config.num_epochs), desc=f"{run_dir.name}"):
        for audio, pose, lengths in tqdm(train_loader):
            global_step += 1
            train_metrics = train_step(model, audio, pose, lengths, criterion, optimizer, scheduler, device)
            total_train_loss += train_metrics['loss']
            total_train_positional_error += train_metrics['positional_error']
            total_train_angular_error += train_metrics['angular_error']
            train_frames += train_metrics['valid_frames']
            if global_step % eval_per_n_steps == 0:
                dev_metrics = evaluate(model, dev_loader, criterion, device)
                train_losses.append(total_train_loss / train_frames)
                train_positional_errors.append(total_train_positional_error / train_frames)
                train_angular_errors.append(total_train_angular_error / train_frames)
                dev_losses.append(dev_metrics['loss'])
                dev_positional_errors.append(dev_metrics['positional_error'])
                dev_angular_errors.append(dev_metrics['angular_error']) 
                
                if WANDB_AVAILABLE:
                    wandb.log({
                        "epoch": epoch + 1,
                        "step": global_step,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "train_loss": total_train_loss / train_frames,
                        "train_positional_error": total_train_positional_error / train_frames,
                        "train_angular_error": total_train_angular_error / train_frames,
                        "dev_loss": dev_metrics['loss'],
                        "dev_positional_error": dev_metrics['positional_error'],
                        "dev_angular_error": dev_metrics['angular_error'],
                    })

                print(f"[{run_dir.name}] Epoch {epoch+1}/{config.num_epochs}")
                print(f"  Train Loss: {total_train_loss / train_frames:.6f}")
                print(f"  Train Positional Error: {total_train_positional_error / train_frames:.4f} m")
                print(f"  Train Angular Error: {total_train_angular_error / train_frames:.4f}°")
                print(f"  Dev Loss: {dev_metrics['loss']:.6f}")
                print(f"  Dev Positional Error: {dev_metrics['positional_error']:.4f} m")
                print(f"  Dev Angular Error: {dev_metrics['angular_error']:.4f}°")

                if dev_metrics['loss'] < best_dev_loss:
                    best_dev_loss = dev_metrics['loss']
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  Saved best model to {best_model_path} (dev loss: {best_dev_loss:.6f})")
                
                train_frames = 0
                total_train_loss = 0.0
                total_train_positional_error = 0.0
                total_train_angular_error = 0.0

    dev_metrics = evaluate(model, dev_loader, criterion, device)
    train_losses.append(total_train_loss / train_frames)
    train_positional_errors.append(total_train_positional_error / train_frames)
    train_angular_errors.append(total_train_angular_error / train_frames)


    dev_losses.append(dev_metrics['loss'])
    dev_positional_errors.append(dev_metrics['positional_error'])
    dev_angular_errors.append(dev_metrics['angular_error']) 
    
    if WANDB_AVAILABLE:
        wandb.log({
            "epoch": epoch + 1,
            "step": global_step,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "train_loss": total_train_loss / train_frames,
            "train_positional_error": total_train_positional_error / train_frames,
            "train_angular_error": total_train_angular_error / train_frames,
            "dev_loss": dev_metrics['loss'],
            "dev_positional_error": dev_metrics['positional_error'],
            "dev_angular_error": dev_metrics['angular_error'],
        })

    print(f"[{run_dir.name}] Epoch {epoch+1}/{config.num_epochs}")
    print(f"  Train Loss: {total_train_loss / train_frames:.6f}")
    print(f"  Train Positional Error: {total_train_positional_error / train_frames:.4f} m")
    print(f"  Train Angular Error: {total_train_angular_error / train_frames:.4f}°")
    print(f"  Dev Loss: {dev_metrics['loss']:.6f}")
    print(f"  Dev Positional Error: {dev_metrics['positional_error']:.4f} m")
    print(f"  Dev Angular Error: {dev_metrics['angular_error']:.4f}°")

    if dev_metrics['loss'] < best_dev_loss:
        best_dev_loss = dev_metrics['loss']
        torch.save(model.state_dict(), best_model_path)
        print(f"  Saved best model to {best_model_path} (dev loss: {best_dev_loss:.6f})")

    # Test with best model
    print(f"[{run_dir.name}] Evaluating best model on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        train_metrics = evaluate(model, train_loader, criterion, device)
        dev_metrics = evaluate(model, dev_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"[{run_dir.name}] Test Loss: {test_metrics['loss']:.6f}")
    print(f"[{run_dir.name}] Test Positional Error: {test_metrics['positional_error']:.4f} m")
    print(f"[{run_dir.name}] Test Angular Error: {test_metrics['angular_error']:.4f}°")

    if WANDB_AVAILABLE:
        wandb.log({
            "train_loss": train_metrics['loss'],
            "train_positional_error": train_metrics['positional_error'],
            "train_angular_error": train_metrics['angular_error'],
            "dev_loss": dev_metrics['loss'],
            "dev_positional_error": dev_metrics['positional_error'],
            "dev_angular_error": dev_metrics['angular_error'],
            "test_loss": test_metrics['loss'],
            "test_positional_error": test_metrics['positional_error'],
            "test_angular_error": test_metrics['angular_error'],
        })
        wandb.finish()

    # Save history & curves into this run directory
    save_training_history(
        train_losses, dev_losses,
        train_positional_errors, train_angular_errors,
        dev_positional_errors, dev_angular_errors,
        train_metrics, dev_metrics, test_metrics,
        save_path=run_dir / "training_history.json"
    )
    plot_training_curves(
        train_losses, dev_losses,
        save_path=run_dir / "training_curve.png"
    )

def main(args):
    # Set random seed for reproducibility
    set_seed(63)
    
    # Configuration
    data_root = Path("data/Main")
    train_dir = data_root / Path("Train")
    dev_dir = data_root / Path("Dev")
    test_dir = data_root / Path("Test")
    
    window_length = args.window_length

    # Create datasets
    print("Loading datasets...")
    train_dataset = AudioPoseDataset(train_dir, window_length)
    dev_dataset = AudioPoseDataset(dev_dir, window_length)
    test_dataset = AudioPoseDataset(test_dir, window_length)
    
    for seq, seqname in ((LSTMSeq, "LSTM"), (TransformerSeq, "Transformer")):
        base_config = {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "dropout": 0.1,
            "num_epochs": 1,
            "feature_extractor": MLPExtractor,
            "sequence_model": seq,
            "head": MLPHead,
        }

        layers = [1, 2, 4, 8]
        # hidden_dims = [64, 128, 256]
        hidden_dims = [512]

        runs_root = Path(f"runs_window_{window_length}")
        for layer in layers:
            for dim in hidden_dims:
                cfg_dict = dict(base_config)
                cfg_dict["num_layers"] = layer
                cfg_dict["hidden_dim"] = dim 
                config = TrainConfig(cfg_dict)

                run_name = f"MLPExtractor_{layer}Layer{seqname}_MLPHead/dim_{dim}"
                run_dir = runs_root / run_name

                run_experiment(config, train_dataset, dev_dataset, test_dataset, run_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--window-length', type=int, default=50)
    args = parser.parse_args()
    main(args)
