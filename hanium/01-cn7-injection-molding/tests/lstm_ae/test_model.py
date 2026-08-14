import torch

from lstm_ae.model import LSTMAutoencoder


def test_forward_pass_preserves_input_shape():
    torch.manual_seed(0)
    model = LSTMAutoencoder(num_features=24, hidden_size=8, latent_dim=4)
    x = torch.randn(5, 12, 24)

    output = model(x)

    assert output.shape == x.shape


def test_forward_pass_works_for_different_batch_and_seq_sizes():
    torch.manual_seed(0)
    model = LSTMAutoencoder(num_features=3, hidden_size=8, latent_dim=4)
    x = torch.randn(2, 7, 3)

    output = model(x)

    assert output.shape == (2, 7, 3)
