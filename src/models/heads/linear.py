# Created Nov 8th, 2025
# Author: Jaduk Suh
import torch
from torch import nn

# Assuming Quaternions
OUTPUT_DIM = 4

class LinearHead(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(in_features=hidden_dim, out_features=OUTPUT_DIM)
    
    def forward(self, x):
        return self.linear(x)