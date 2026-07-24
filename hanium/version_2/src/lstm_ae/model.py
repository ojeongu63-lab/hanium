import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64, latent_dim: int = 16):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_size=num_features, hidden_size=hidden_size, batch_first=True
        )
        self.to_latent = nn.Linear(hidden_size, latent_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim, hidden_size=hidden_size, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.to_latent(h_n[-1])
        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(repeated)
        return self.output_layer(decoded)
