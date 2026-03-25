"""Bidirectional LSTM utilities for observation-window to full-state estimation.

This module implements a bidirectional LSTM for state estimation from observation windows.
In our context, BiLSTM processes a fixed window of observations (e.g., 5 time steps) to 
estimate the state at the final time step. The "bidirectional" aspect allows the model
to consider information from both earlier and later observations within the window,
which is analogous to "smoothing" in data assimilation terminology.

Key considerations for time series state estimation:
1. We use BiLSTM on fixed windows, not for real-time forecasting
2. "Future" information refers to later observations within the known window
3. This is similar to smoothing vs filtering in data assimilation
4. The approach is valid when we have a complete observation window available

References:
1) Graves, A., Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM 
   and other neural network architectures. Neural Networks, 18(5-6), 602-610.
2) Schuster, M., Paliwal, K. K. (1997). Bidirectional recurrent neural networks. 
   IEEE Transactions on Signal Processing, 45(11), 2673-2681.
3) Brajard, J., et al. (2020). Combining data assimilation and machine learning to emulate 
   a dynamical model from sparse and noisy observations. J. Computational Science, 44, 101171.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    raise RuntimeError("PyTorch is required. Install torch first.") from e


def make_windows(
    Y: np.ndarray,
    X: Optional[np.ndarray],
    window: int,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Create sliding windows from observation sequences."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be 2D: [T, obs_dim]")
    if window < 1:
        raise ValueError("window must be >= 1")

    X_arr: Optional[np.ndarray]
    if X is None:
        X_arr = None
    else:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D: [T, state_dim]")

    T = int(Y.shape[0])
    if T < window:
        raise ValueError("not enough samples for the given window")
    if X_arr is not None:
        if int(X_arr.shape[0]) != T:
            raise ValueError("X and Y must have the same length (T)")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    idx: list[int] = []
    for k in range(window - 1, T):
        xs.append(Y[k - window + 1 : k + 1, :])
        idx.append(k)
        if X_arr is not None:
            ys.append(X_arr[k, :])

    Xw = np.stack(xs, axis=0)
    yw = np.stack(ys, axis=0) if X_arr is not None else None
    return Xw, yw, np.asarray(idx, dtype=int)


@dataclass(frozen=True)
class Standardizer:
    """Data standardization utility."""
    mean: np.ndarray
    std: np.ndarray

    def transform(self, a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, dtype=float) - self.mean) / self.std

    def inverse(self, a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=float) * self.std + self.mean


def fit_standardizers(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Standardizer, Standardizer]:
    """Fit standardizers for input and output data."""
    x_mean = X_train.mean(axis=(0, 1), keepdims=True)
    x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-8
    return Standardizer(x_mean, x_std), Standardizer(y_mean, y_std)


def device_from_arg(device: str) -> torch.device:
    """Parse device argument."""
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


class ObsToStateBiLSTM(nn.Module):
    """Bidirectional LSTM for observation-to-state mapping.
    
    This model processes observation windows in both forward and backward directions,
    then combines information from both directions to predict the full state at the
    final time step. This is particularly useful for state estimation where we have
    a complete window of observations and want to leverage all available information.
    
    Architecture:
    - Bidirectional LSTM processes the observation sequence
    - Final hidden states from both directions are concatenated
    - Linear layer maps the concatenated features to state space
    
    Note: This approach is valid for "smoothing" scenarios where we have access to
    a complete observation window, as opposed to real-time "filtering" scenarios.
    """
    
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hidden = hidden
        self.n_layers = n_layers

        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,  # Key difference: bidirectional=True
        )
        
        # Output layer: hidden_size * 2 because bidirectional
        self.out = nn.Linear(hidden * 2, state_dim)

    def forward(self, y: torch.Tensor, mask=None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            y: Input observations [batch_size, seq_len, obs_dim]
            mask: Optional mask (not used in current implementation)
            
        Returns:
            Predicted states [batch_size, state_dim]
        """
        # BiLSTM output and final hidden states
        _, (h_n, _) = self.rnn(y)
        
        # Extract final hidden states from both directions of the last layer
        # For bidirectional LSTM with n_layers:
        # h_n shape: [n_layers * 2, batch_size, hidden_size]
        # Last layer forward:  h_n[-2, :, :]  (second to last)
        # Last layer backward: h_n[-1, :, :]  (last)
        
        forward_final = h_n[-2, :, :]   # [batch_size, hidden_size] - forward direction final state
        backward_final = h_n[-1, :, :]  # [batch_size, hidden_size] - backward direction final state
        
        # Concatenate both directions to get full bidirectional representation
        # This ensures we use the complete information from both directions:
        # - Forward: has seen the entire sequence from start to end
        # - Backward: has seen the entire sequence from end to start
        final_features = torch.cat([forward_final, backward_final], dim=1)  # [batch_size, hidden_size * 2]
        
        return self.out(final_features)


def train_obs_to_state_bilstm(
    Y: np.ndarray,
    X_true: np.ndarray,
    *,
    window: int,
    hidden: int = 128,
    n_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 200,
    batch_size: int = 256,
    train_frac: float = 0.9,
    val_frac: float = 0.1,
    patience: int = 20,
    grad_clip: float = 1.0,
    seed: int = 0,
    device: str = "auto",
) -> dict:
    """Train a bidirectional LSTM model.
    
    Args:
        Y: Observation sequences [T, obs_dim]
        X_true: True states [T, state_dim]
        window: Sequence window length
        hidden: Hidden dimension
        n_layers: Number of LSTM layers
        dropout: Dropout rate
        lr: Learning rate
        weight_decay: Weight decay
        epochs: Maximum training epochs
        batch_size: Batch size
        train_frac: Training data fraction
        val_frac: Validation data fraction (from training data)
        patience: Early stopping patience
        grad_clip: Gradient clipping threshold
        seed: Random seed
        device: Device ('auto', 'cpu', 'cuda')
        
    Returns:
        Dictionary containing trained model and metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create windowed data
    Xw, yw, idx = make_windows(Y, X_true, window=window)
    if yw is None:
        raise RuntimeError("internal error: yw should not be None")

    # Split data
    n_samples = Xw.shape[0]
    n_train = max(1, int(train_frac * n_samples))
    X_train = Xw[:n_train]
    y_train = yw[:n_train]
    X_test = Xw[n_train:]
    y_test = yw[n_train:]
    idx_test = idx[n_train:]

    # Further split training data for validation
    n_train_samples = X_train.shape[0]
    if n_train_samples <= 1:
        n_val = 0
        n_train_actual = n_train_samples
    else:
        n_val = int(val_frac * n_train_samples)
        n_val = max(1, min(n_val, n_train_samples - 1))
        n_train_actual = n_train_samples - n_val

    X_val = X_train[n_train_actual:]
    y_val = y_train[n_train_actual:]
    X_train = X_train[:n_train_actual]
    y_train = y_train[:n_train_actual]

    # Standardize data
    x_scaler, y_scaler = fit_standardizers(X_train, y_train)
    X_train_n = x_scaler.transform(X_train)
    X_val_n = x_scaler.transform(X_val)
    X_test_n = x_scaler.transform(X_test)
    y_train_n = y_scaler.transform(y_train)
    y_val_n = y_scaler.transform(y_val)
    y_test_n = y_scaler.transform(y_test)

    has_val = X_val_n.shape[0] > 0

    # Initialize model
    dev = device_from_arg(device)
    model = ObsToStateBiLSTM(
        obs_dim=Y.shape[-1],
        state_dim=X_true.shape[-1],
        hidden=hidden,
        n_layers=n_layers,
        dropout=dropout,
    ).to(dev)

    # Create data loaders
    train_ds = TensorDataset(torch.from_numpy(X_train_n).float(), torch.from_numpy(y_train_n).float())
    test_ds = TensorDataset(torch.from_numpy(X_test_n).float(), torch.from_numpy(y_test_n).float())
    val_ds = (
        TensorDataset(torch.from_numpy(X_val_n).float(), torch.from_numpy(y_val_n).float()) if has_val else None
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False) if val_ds is not None else None
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Initialize optimizer and loss
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    history_train: list[float] = []
    history_val: list[float] = []

    for _epoch in range(epochs):
        # Training phase
        model.train()
        tr_loss = 0.0
        tr_count = 0
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            pred = model(xb, mask=None)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_count += xb.size(0)
        tr_loss = tr_loss / tr_count if tr_count > 0 else float("inf")

        # Validation phase
        if val_loader is None:
            va_loss = tr_loss
        else:
            model.eval()
            va_loss = 0.0
            va_count = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(dev)
                    yb = yb.to(dev)
                    pred = model(xb, mask=None)
                    loss = loss_fn(pred, yb)
                    va_loss += loss.item() * xb.size(0)
                    va_count += xb.size(0)
            va_loss = va_loss / va_count if va_count > 0 else tr_loss

        history_train.append(tr_loss)
        history_val.append(va_loss)

        # Early stopping
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)

    # Evaluate on test set
    model.eval()
    preds_n: list[np.ndarray] = []
    trues_n: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            preds_n.append(model(xb, mask=None).cpu().numpy())
            trues_n.append(yb.cpu().numpy())
    pred_n = np.concatenate(preds_n, axis=0) if len(preds_n) else np.zeros((0, X_true.shape[-1]))
    true_n = np.concatenate(trues_n, axis=0) if len(trues_n) else np.zeros((0, X_true.shape[-1]))

    # Inverse transform predictions
    pred = y_scaler.inverse(pred_n)
    true = y_scaler.inverse(true_n)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((pred - true) ** 2)) if pred.size else float("nan")

    return {
        "model": model,
        "device": str(dev),
        "x_mean": np.asarray(x_scaler.mean, dtype=float),
        "x_std": np.asarray(x_scaler.std, dtype=float),
        "y_mean": np.asarray(y_scaler.mean, dtype=float),
        "y_std": np.asarray(y_scaler.std, dtype=float),
        "window": window,
        "hidden": hidden,
        "n_layers": n_layers,
        "dropout": dropout,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
        "n_test": X_test.shape[0],
        "idx_test": np.asarray(idx_test, dtype=int),
        "rmse_test": rmse,
        "train_loss": np.asarray(history_train, dtype=float),
        "val_loss": np.asarray(history_val, dtype=float),
    }