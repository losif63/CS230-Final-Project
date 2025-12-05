# Created Nov 8th, 2025
# Author: Jaduk Suh
import torch, torchaudio
import os
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

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

    
    def forward(self, x, lengths):
        # LSTM returns (output, (h_n, c_n)), we only need the final state h_n
        if lengths is not None:
            # pack to ignore padded timesteps
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)
        # h_n: [num_layers, B, hidden_dim]
        last_layer_h = h_n[-1]               # [B, hidden_dim]
        
        # LayerNorm normalizes over the last dimension (hidden_dim) at each timestep
        last_layer_h = self.layer_norm(last_layer_h)
        return last_layer_h