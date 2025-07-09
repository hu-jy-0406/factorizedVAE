import torch
import torch.nn as nn
import math

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            -math.log(10000.0) * torch.arange(0, embedding_dim, 2).float() / embedding_dim
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x shape: (B, L, d_model)
        returns x + positional embedding
        """
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0).to(x.device)