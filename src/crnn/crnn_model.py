"""
Author: Prerana Rane
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class CRNNRotation(nn.Module):
    def __init__(self,
                 n_channels: int = 6,
                 n_fft: int = 1024,
                 hop_length: int = 240,  # 5ms hop for 48kHz
                 output_dim: int = 7,  # 3 position + 4 quaternion
                 cnn_channels: List[int] = [32, 64, 128],
                 rnn_hidden_size: int = 256,
                 rnn_num_layers: int = 2,
                 dropout: float = 0.3):

        super().__init__()
        
        self.n_channels = n_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.output_dim = output_dim
        
        self.stft_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None,  # Return complex values
            normalized=False,
            window_fn=torch.hann_window
        )
        
        self.n_freq_bins = n_fft // 2 + 1        
        cnn_input_channels = 2 * n_channels        
        cnn_layers = []
        in_ch = cnn_input_channels
        
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout)
            ])
            in_ch = out_ch
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.freq_reduced = self.n_freq_bins // (2 ** len(cnn_channels))
        self.rnn_input_dim = cnn_channels[-1] * self.freq_reduced
        
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout if rnn_num_layers > 1 else 0,
            bidirectional=True
        )
        
        fc_input_dim = rnn_hidden_size * 2  # *2 for bidirectional
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        batch_size, n_frames, n_channels, n_samples = x.shape
        
        frame_features = []

        for frame_idx in range(n_frames):
            frame = x[:, frame_idx, :, :]

            stft_features = []
            for ch_idx in range(n_channels):
                audio_ch = frame[:, ch_idx, :]
                stft_ch = self.stft_transform(audio_ch)
                stft_features.append(stft_ch.real)
                stft_features.append(stft_ch.imag)

            stft_frame = torch.stack(stft_features, dim=1)

            cnn_out = self.cnn(stft_frame)

            cnn_out = cnn_out.mean(dim=-1)

            cnn_out = cnn_out.view(batch_size, -1)
            frame_features.append(cnn_out)

        x = torch.stack(frame_features, dim=1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        pose = self.fc(x)

        position = pose[:, :3]
        quaternion = pose[:, 3:]
        quaternion = F.normalize(quaternion, p=2, dim=1)

        return torch.cat([position, quaternion], dim=1)


def create_model(config):
    model = CRNNRotation(
        n_channels=config.N_CHANNELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        cnn_channels=config.CNN_CHANNELS,
        rnn_hidden_size=config.RNN_HIDDEN_SIZE,
        rnn_num_layers=config.RNN_NUM_LAYERS,
        dropout=config.DROPOUT
    )
    model = model.to(config.DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Model created with {num_params:,} trainable parameters")

    return model
