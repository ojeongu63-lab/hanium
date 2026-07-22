import numpy as np
import torch

from lstm_ae.model import LSTMAutoencoder
from lstm_ae.training import train_autoencoder


def test_train_autoencoder_reduces_loss_on_easy_synthetic_target():
    torch.manual_seed(0)
    train_windows = np.zeros((20, 4, 3), dtype=np.float32)
    model = LSTMAutoencoder(num_features=3, hidden_size=8, latent_dim=4)

    losses = train_autoencoder(
        model, train_windows, epochs=20, batch_size=4, learning_rate=1e-2
    )

    assert len(losses) == 20
    assert all(np.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]
