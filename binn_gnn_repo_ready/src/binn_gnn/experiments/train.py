from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    ys: list[np.ndarray] = []
    logits: list[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        ys.append(yb.detach().cpu().numpy())
        logits.append(out.detach().cpu().numpy())

    y = np.concatenate(ys, axis=0)
    logit = np.concatenate(logits, axis=0)
    return y, logit


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    # logits: [N, C]
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = probs.argmax(axis=1)

    metrics: Dict[str, float] = {}
    metrics["balanced_acc"] = float(balanced_accuracy_score(y_true, pred))

    if probs.shape[1] == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    val_metrics: Dict[str, float]


def train_classifier(
    *,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 10,
    verbose: bool = True,
) -> TrainResult:
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        it = train_loader
        if verbose:
            it = tqdm(it, desc=f"epoch {epoch}/{epochs}")

        for xb, yb in it:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        # Validation loss
        model.eval()
        val_running = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                val_running += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)

        val_loss = val_running / max(1, val_n)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if verbose:
            print(f"epoch {epoch:03d} train_loss={running/max(1,n):.4f} val_loss={val_loss:.4f} best={best_val_loss:.4f}")

        if patience >= early_stopping_patience:
            if verbose:
                print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_val, logits_val = predict_logits(model, val_loader, device)
    val_metrics = compute_metrics(y_val, logits_val)

    return TrainResult(best_epoch=best_epoch, best_val_loss=best_val_loss, val_metrics=val_metrics)
