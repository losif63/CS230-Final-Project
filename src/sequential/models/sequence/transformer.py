# Created Nov 8th, 2025
# Author: Jaduk Suh
import torch, torchaudio
import os
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        B, T, D = x.shape
        x = x + self.pe[:, :T, :]
        return self.dropout(x)

class TransformerSeq(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, nhead: int = 4,
                 max_len: int = 5000, *args, **kwargs):
        """
        Args:
            hidden_dim: d_model (must be divisible by nhead)
            num_layers: number of Transformer encoder layers
            dropout: dropout used inside the encoder layers
            nhead: number of attention heads (default 4)
            max_len: max sequence length for positional encoding
        """
        super().__init__(*args, **kwargs)
        assert hidden_dim % nhead == 0, "hidden_dim must be divisible by nhead"

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.nhead = nhead

        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=0.0, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,   # so we can use (B, T, D)
            activation="relu",
            norm_first=False,   # standard PyTorch default
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, hidden_dim)  - padded sequences
            lengths: (B,)          - true lengths (number of valid frames) per sequence
            return_sequence: if True, return full (B, T, H) sequence,
                             else return pooled last state (B, H)

        Returns:
            (B, hidden_dim)  - last valid token representation
        """
        B, T, D = x.shape

        if lengths is not None:
            # True where we want to ignore positions (PAD)
            src_key_padding_mask = (
                torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            )  # (B, T)
        else:
            # fallback: infer pad from zeros (less ideal)
            src_key_padding_mask = (x.abs().sum(dim=-1) < 1e-6)

        # Add positional encoding
        x = self.pos_encoding(x)  # (B, T, D)

        # Transformer encoder with padding mask
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, D)

        # LayerNorm
        out = self.layer_norm(out)

        # ---- Pool: take last *valid* token per sequence ----
        # lengths is count, so last index = lengths - 1
        last_idx = lengths - 1  # (B,)

        # gather: out[b, last_idx[b], :]
        pooled = out[torch.arange(B, device=x.device), last_idx]  # (B, D)
        return pooled