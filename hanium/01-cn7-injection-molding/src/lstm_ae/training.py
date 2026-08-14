import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_autoencoder(
    model: nn.Module,
    train_windows: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> list[float]:
    x = torch.tensor(train_windows, dtype=torch.float32)
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        epoch_loss = total_loss / len(dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")
    return epoch_losses
