import math
from time import perf_counter

import torch
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_function: _Loss,
    *,
    gradient_clip_norm: float | None = None,
) -> dict[str, int | float]:
    if gradient_clip_norm is not None and (
        isinstance(gradient_clip_norm, bool)
        or not math.isfinite(gradient_clip_norm)
        or gradient_clip_norm <= 0
    ):
        raise ValueError("gradient_clip_norm must be finite and positive")
    model.train()
    device = next(model.parameters()).device

    started = perf_counter()
    running_loss = 0.0
    target_count = 0

    for contexts, targets in data_loader:
        contexts, targets = contexts.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(contexts)
        loss = loss_function(logits, targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Training loss is not finite")

        loss.backward()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm, error_if_nonfinite=True
            )
        optimizer.step()

        batch_targets = targets.numel()
        running_loss += loss.item() * batch_targets
        target_count += batch_targets

    if target_count == 0:
        raise ValueError("Training loader contains no targets")

    return {
        "loss": running_loss / target_count,
        "target_count": target_count,
        "elapsed_seconds": perf_counter() - started,
    }
