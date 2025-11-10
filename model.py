from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    def __init__(self, 
                 n_channels: int = 6, 
                 samples_per_frame: int = 2400,
                 hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 7, 
                 dropout: float = 0.3):
        """
        Inputs:
            n_channels: Number of audio channels
            samples_per_frame: Number of audio samples per frame
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (7 for 6DOF: 3 position + 4 quaternion)
            dropout: Dropout probability
        """
        super().__init__()

        self.n_channels = n_channels
        self.samples_per_frame = samples_per_frame
        self.output_dim = output_dim
        input_dim = n_channels * samples_per_frame

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        #x: (batch, n_channels, samples_per_frame) raw audio

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        pose = self.network(x)

        # Normalize quaternion part (last 4 values)
        position = pose[:, :3]
        quaternion = pose[:, 3:]
        quaternion = F.normalize(quaternion, p=2, dim=1)

        return torch.cat([position, quaternion], dim=1)


def create_model(config):
    model = MLPBaseline(
        n_channels=config.N_CHANNELS,
        samples_per_frame=config.SAMPLES_PER_FRAME,
        hidden_dims=config.HIDDEN_DIMS,
        output_dim=config.OUTPUT_DIM,
        dropout=config.DROPOUT
    )
    model = model.to(config.DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Model created with {num_params:,} trainable parameters")
    
    return model
