import torch
import torch.nn as nn

class PositionEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super().__init__()
        position_encoding = torch.zeros(max_length, embedding_dim)
        positions = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        position_encoding[:, 0::2] = torch.sin(positions * div_term)
        position_encoding[:, 1::2] = torch.cos(positions * div_term)
        position_encoding = position_encoding.unsqueeze(0)
        self.register_buffer('pe', position_encoding)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CharacterTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=4, num_layers=2, ff_dim=512, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionEncoding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, ff_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_seq):
        return self.output_layer(self.transformer_encoder(self.pos_encoder(self.embedding(input_seq)))[:, -1, :]) 