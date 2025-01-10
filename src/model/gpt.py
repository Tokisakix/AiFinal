import torch
import torch.nn as nn

from .positionalencoding import PositionalEncoding
from .gptblock import GPTBlock

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, d_ff=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.out = nn.Linear(d_model, vocab_size)
        return

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)

        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, mask)

        x = x.transpose(0, 1)

        return self.out(x)