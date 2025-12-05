"""
CRNN Head Rotation Estimator for Smart Glasses
Author: Prerana Rane
Date: November 2025
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
        batch_size, n_channels, n_samples = x.shape
        
        stft_features = []
        for ch_idx in range(n_channels):
            audio_ch = x[:, ch_idx, :]
            stft_ch = self.stft_transform(audio_ch)            
            stft_real = stft_ch.real  
            stft_imag = stft_ch.imag  
            stft_features.append(stft_real)
            stft_features.append(stft_imag)
        
        x = torch.stack(stft_features, dim=1)        
        x = self.cnn(x)  
        
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2)  
        x = x.contiguous().view(batch, time, -1) 
        x, _ = self.rnn(x)          
        x = x[:, -1, :]  
        
        pose = self.fc(x)  
        
        position = pose[:, :3]
        quaternion = pose[:, 3:]
        quaternion = F.normalize(quaternion, p=2, dim=1)
        
        return torch.cat([position, quaternion], dim=1)
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    torch.manual_seed(42)
    
    # Initialize model
    model = CRNNRotation(
        n_channels=6,
        n_fft=1024,
        hop_length=240,
        cnn_channels=[32, 64, 128],
        rnn_hidden_size=256,
        rnn_num_layers=2,
        dropout=0.3
    )
    
    print(f"Model initialized with {model.get_num_parameters():,} parameters")

if __name__ == "__main__":
    main()