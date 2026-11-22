"""Frozen encoder embedding and linear-probe evaluation."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .train import _autocast, pool


def embed(encoder: torch.nn.Module, x: np.ndarray, batch: int = 512,
          device: str = "cuda") -> np.ndarray:
    """Return frozen pooled embeddings as float32 ``[n, 2d]`` numpy values."""
    if batch < 1:
        raise ValueError("batch must be positive")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("x must have shape [n, L]")
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    encoder = encoder.to(dev)
    encoder.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.as_tensor(x[start:start + batch], dtype=torch.float32, device=dev)
            with _autocast(dev):
                tokens = encoder(xb, hide=None)
                features = pool(tokens)
            chunks.append(features.float().cpu().numpy())
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def linear_probe(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray,
                 yte: np.ndarray, kind: str) -> dict:
    """Fit a standardized logistic-regression or ridge probe and report metrics."""
    Ztr = np.asarray(Ztr, dtype=np.float64)
    Zte = np.asarray(Zte, dtype=np.float64)
    ytr = np.asarray(ytr)
    yte = np.asarray(yte)
    if Ztr.ndim != 2 or Zte.ndim != 2 or Ztr.shape[1] != Zte.shape[1]:
        raise ValueError("Ztr and Zte must be 2D arrays with the same feature width")
    scaler = StandardScaler()
    train_features = scaler.fit_transform(Ztr)
    test_features = scaler.transform(Zte)
    if kind == "cls":
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        estimator = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)
        search = GridSearchCV(estimator, {"C": np.logspace(-3, 2, 6)}, cv=folds,
                              scoring="accuracy", refit=True, n_jobs=1)
        search.fit(train_features, ytr)
        prediction = search.predict(test_features)
        return {"metric": float(np.mean(prediction == yte)), "C": float(search.best_params_["C"])}
    if kind == "reg":
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
        model.fit(train_features, ytr.astype(np.float64))
        prediction = np.asarray(model.predict(test_features), dtype=np.float64)
        residual = np.square(yte.astype(np.float64) - prediction).sum()
        centered = yte.astype(np.float64) - yte.astype(np.float64).mean()
        total = np.square(centered).sum()
        r_squared = float(1.0 - residual / total) if total > 0 else float("nan")
        rho = float(spearmanr(yte, prediction).statistic)
        return {"metric": r_squared, "spearman": rho, "alpha": float(model.alpha_)}
    raise ValueError("kind must be 'cls' or 'reg'")
