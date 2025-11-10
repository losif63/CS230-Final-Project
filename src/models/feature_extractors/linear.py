# Created Nov 8th, 2025
# Author: Jaduk Suh
import torch, torchaudio
from torch import nn

class LinearExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.batch_norm = nn.BatchNorm1d(num_features=self.hidden_dim)
    
    def forward(self, x):
        x = self.linear(x)
        return self.batch_norm(x)
