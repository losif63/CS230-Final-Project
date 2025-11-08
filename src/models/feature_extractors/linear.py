# Created Nov 8th, 2025
# Author: Jaduk Suh
import torch, torchaudio
from torch import nn

SAMPLE_RATE = 48000
FRAME_LEN = 0.05

class LinearExtractor(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(int(SAMPLE_RATE * FRAME_LEN), self.hidden_dim, bias=False)
        self.batch_norm = nn.BatchNorm1d(num_features=self.hidden_dim)
    
    def forward(self, x):
        x = self.linear(x)
        return self.batch_norm(x)
