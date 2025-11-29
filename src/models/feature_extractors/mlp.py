# Created Nov 27th, 2025
# Author: Jaduk Suh
import torch, torchaudio
from torch import nn

class MLPExtractor2L(nn.Module):
    def __init__(self, input_dim, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim * 2)
        self.layernorm = nn.LayerNorm(self.hidden_dim * 2)
        self.linear2 = nn.Linear(self.hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.layernorm(x)
        x = self.linear2(x)
        return x