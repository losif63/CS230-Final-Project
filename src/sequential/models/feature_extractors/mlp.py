# Created Nov 27th, 2025
# Author: Jaduk Suh
import torch, torchaudio
from torch import nn

class MLPExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.linear = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_dim * num_channels, self.hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
    
    def forward(self, x):
        batch_size, seq_len, num_channels, _ = x.shape
        x = x.view(-1, self.input_dim)
        x = self.linear(x)
        x = self.relu(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.linear2(x)
        return self.layer_norm(x)