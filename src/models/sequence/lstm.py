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
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    
    def forward(self, x):
        # LSTM returns (output, (h_n, c_n)), we only need the output
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_dim)
        # LayerNorm normalizes over the last dimension (hidden_dim) at each timestep
        lstm_out = self.layer_norm(lstm_out)
        return lstm_out