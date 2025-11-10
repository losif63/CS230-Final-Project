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
        # LSTM returns (output, (h_n, c_n)), we only need the output
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_dim)
        # BatchNorm1d expects (batch, features, seq_len) or (batch, features)
        # We need to transpose to (batch, hidden_dim, seq_len) for BatchNorm1d
        lstm_out = lstm_out.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        lstm_out = self.batch_norm(lstm_out)
        # Transpose back to (batch, seq_len, hidden_dim)
        return lstm_out.transpose(1, 2)