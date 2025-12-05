# Created Nov 29th, 2025
# Author: Jaduk Suh
import torch
from torch import nn

# Position (3) + Quaternion (4) = 7 values
LEVEL1 = 64
LEVEL2 = 32
OUTPUT_DIM = 7

class MLPHead(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_features=hidden_dim, out_features=LEVEL1)
        self.layernorm1 = nn.LayerNorm(LEVEL1)
        self.linear2 = nn.Linear(in_features=LEVEL1, out_features=LEVEL2)
        self.layernorm2 = nn.LayerNorm(LEVEL2)
        self.linear3 = nn.Linear(in_features=LEVEL2, out_features=OUTPUT_DIM, bias=True)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.layernorm1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.layernorm2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x