# Created Nov 8th, 2025
# Author: Jaduk Suh
import torch, torchaudio
import os
from torch import nn

class LSTMSeq(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=num_layers,
            bias=False,
            batch_first=True,
            dropout=dropout
        )
        self.batch_norm = nn.BatchNorm1d(num_features=self.hidden_dim)

    
    def forward(self, x):
        x = self.lstm(x)
        return self.batch_norm(x)