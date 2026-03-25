"""LSTM utilities for observation-window to full-state estimation.

References
1) Hochreiter, S., Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
   DOI: 10.1162/neco.1997.9.8.1735
2) Gers, F. A., Schmidhuber, J., Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM.
   Neural Computation, 12(10), 2451-2471. DOI: 10.1162/089976600300015015
3) Hassanzadeh, P., Subramanian, D. (2020). Data-driven predictions of a multiscale Lorenz 96 chaotic system
   using machine-learning methods: reservoir computing, artificial neural network, and long short-term memory
   network. Nonlinear Processes in Geophysics, 27, 373-389. DOI: 10.5194/npg-27-373-2020
4) Brajard, J., Carrassi, A., Bocquet, M., Bertino, L. (2020). Combining data assimilation and machine learning
   to emulate a dynamical model from sparse and noisy observations: A case study with the Lorenz 96 model.
   Journal of Computational Science, 44, 101171. DOI: 10.1016/j.jocs.2020.101171
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
    x_mean = X_train.mean(axis=(0, 1), keepdims=True)
    x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-8
    return Standardizer(x_mean, x_std), Standardizer(y_mean, y_std)


def device_from_arg(device: str) -> torch.device:
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


class ObsToStateLSTM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)

        self.rnn = nn.LSTM(
            input_size=int(obs_dim),
            hidden_size=int(hidden),
            num_layers=int(n_layers),
            batch_first=True,
            dropout=float(dropout) if int(n_layers) > 1 else 0.0,
        )
        self.out = nn.Linear(int(hidden), int(state_dim))

    def forward(self, y: torch.Tensor, mask=None) -> torch.Tensor:
        x, _ = self.rnn(y)
        x = x[:, -1, :]
        return self.out(x)


def train_obs_to_state_lstm(
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
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    Xw, yw, idx = make_windows(Y, X_true, window=int(window))
    if yw is None:
        raise RuntimeError("internal error: yw should not be None")

    n_samples = int(Xw.shape[0])
    n_train = max(1, int(float(train_frac) * n_samples))
    X_train = Xw[:n_train]
    y_train = yw[:n_train]
    X_test = Xw[n_train:]
    y_test = yw[n_train:]
    idx_test = idx[n_train:]

    n_train_samples = int(X_train.shape[0])
    if n_train_samples <= 1:
        n_val = 0
        n_train_actual = n_train_samples
    else:
        n_val = int(float(val_frac) * n_train_samples)
        n_val = max(1, min(n_val, n_train_samples - 1))
        n_train_actual = n_train_samples - n_val

    X_val = X_train[n_train_actual:]
    y_val = y_train[n_train_actual:]
    X_train = X_train[:n_train_actual]
    y_train = y_train[:n_train_actual]

    x_scaler, y_scaler = fit_standardizers(X_train, y_train)
    X_train_n = x_scaler.transform(X_train)
    X_val_n = x_scaler.transform(X_val)
    X_test_n = x_scaler.transform(X_test)
    y_train_n = y_scaler.transform(y_train)
    y_val_n = y_scaler.transform(y_val)
    y_test_n = y_scaler.transform(y_test)

    has_val = bool(int(X_val_n.shape[0]) > 0)

    dev = device_from_arg(device)
    model = ObsToStateLSTM(
        obs_dim=int(Y.shape[-1]),
        state_dim=int(X_true.shape[-1]),
        hidden=int(hidden),
        n_layers=int(n_layers),
        dropout=float(dropout),
    ).to(dev)

    train_ds = TensorDataset(torch.from_numpy(X_train_n).float(), torch.from_numpy(y_train_n).float())
    test_ds = TensorDataset(torch.from_numpy(X_test_n).float(), torch.from_numpy(y_test_n).float())
    val_ds = (
        TensorDataset(torch.from_numpy(X_val_n).float(), torch.from_numpy(y_val_n).float()) if has_val else None
    )

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
    val_loader = (
        DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, drop_last=False) if val_ds is not None else None
    )
    test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    history_train: list[float] = []
    history_val: list[float] = []

    for _epoch in range(int(epochs)):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            tr_loss += float(loss.item()) * int(xb.size(0))
            tr_count += int(xb.size(0))
        tr_loss = tr_loss / tr_count if tr_count > 0 else float("inf")

        if val_loader is None:
            va_loss = float(tr_loss)
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
                    va_loss += float(loss.item()) * int(xb.size(0))
                    va_count += int(xb.size(0))
            va_loss = va_loss / va_count if va_count > 0 else float(tr_loss)

        history_train.append(float(tr_loss))
        history_val.append(float(va_loss))

        if va_loss < best_val_loss:
            best_val_loss = float(va_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(dev)

    model.eval()
    preds_n: list[np.ndarray] = []
    trues_n: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            preds_n.append(model(xb, mask=None).cpu().numpy())
            trues_n.append(yb.cpu().numpy())
    pred_n = np.concatenate(preds_n, axis=0) if len(preds_n) else np.zeros((0, int(X_true.shape[-1])))
    true_n = np.concatenate(trues_n, axis=0) if len(trues_n) else np.zeros((0, int(X_true.shape[-1])))

    pred = y_scaler.inverse(pred_n)
    true = y_scaler.inverse(true_n)

    rmse = float(np.sqrt(np.mean((pred - true) ** 2))) if pred.size else float("nan")

    return {
        "model": model,
        "device": str(dev),
        "x_mean": np.asarray(x_scaler.mean, dtype=float),
        "x_std": np.asarray(x_scaler.std, dtype=float),
        "y_mean": np.asarray(y_scaler.mean, dtype=float),
        "y_std": np.asarray(y_scaler.std, dtype=float),
        "window": int(window),
        "hidden": int(hidden),
        "n_layers": int(n_layers),
        "dropout": float(dropout),
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "idx_test": np.asarray(idx_test, dtype=int),
        "rmse_test": float(rmse),
        "train_loss": np.asarray(history_train, dtype=float),
        "val_loss": np.asarray(history_val, dtype=float),
    }