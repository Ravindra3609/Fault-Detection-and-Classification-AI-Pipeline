"""
autoencoder.py
==============
PyTorch Autoencoder for unsupervised anomaly detection.
- Trained only on normal (non-fault) data
- Reconstruction error = anomaly score
- High error → point looks unlike normal data → potential fault
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Optional

MODEL_PATH = Path("models/autoencoder.pt")


class Autoencoder(nn.Module):
    """
    Symmetric encoder-decoder with batch normalization and dropout.
    Architecture scales with input dimension.
    """

    def __init__(self, input_dim: int, latent_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        h1 = max(input_dim // 2, latent_dim * 4)
        h2 = max(input_dim // 4, latent_dim * 2)

        # Encoder: input → h1 → h2 → latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, latent_dim),
            nn.ReLU(),
        )

        # Decoder: latent → h2 → h1 → input (mirror)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error."""
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=1)

    def per_feature_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-feature reconstruction error — shows which sensors are anomalous."""
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2)


def train_autoencoder(
    X_normal: np.ndarray,
    input_dim: Optional[int] = None,
    latent_dim: int = 16,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    dropout: float = 0.2,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple["Autoencoder", list]:
    """
    Train autoencoder on normal (non-fault) data only.
    Returns trained model and loss history.
    """
    if input_dim is None:
        input_dim = X_normal.shape[1]

    model = Autoencoder(input_dim, latent_dim, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    X_tensor = torch.FloatTensor(X_normal).to(device)
    dataset   = TensorDataset(X_tensor)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        avg_loss = epoch_loss / len(X_normal)
        history.append(avg_loss)
        scheduler.step(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss: {avg_loss:.6f}  lr: {optimizer.param_groups[0]['lr']:.2e}")

    return model, history


def anomaly_scores(
    model: "Autoencoder",
    X: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Return per-sample reconstruction error (anomaly score)."""
    model.eval()
    X_t = torch.FloatTensor(X).to(device)
    return model.reconstruction_error(X_t).cpu().numpy()


def anomaly_threshold(scores_normal: np.ndarray, percentile: float = 95.0) -> float:
    """
    Compute anomaly threshold from normal data scores.
    Points above this threshold are flagged as anomalies.
    95th percentile → ~5% false alarm rate on normal data.
    """
    return float(np.percentile(scores_normal, percentile))


def save_autoencoder(model: "Autoencoder", path: Path = MODEL_PATH):
    path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)


def load_autoencoder(input_dim: int, latent_dim: int = 16,
                      path: Path = MODEL_PATH, device: str = "cpu") -> "Autoencoder":
    model = Autoencoder(input_dim, latent_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
