from __future__ import annotations
import random
from typing import List, Dict, Any, Optional, Type

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR

from model import StartEndInsideModel
from data import CoAMTokenDataset, collate_fn
from evaluate import evaluate_exact
from config import DEVICE, BATCH_SIZE, EPOCHS, LR, GRAD_ACCUM, PATIENCE, WEIGHT_DECAY, MODEL_NAME


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def token_loss(pred: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy(pred, gold, reduction='none')
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1)


def train_model(model: nn.Module, train_loader: DataLoader,
                dev_dataset: Optional[CoAMTokenDataset] = None,
                epochs: int = EPOCHS, lr: float = LR, weight_decay: float = WEIGHT_DECAY,
                grad_accum: int = GRAD_ACCUM, patience: int = PATIENCE,
                device: str = DEVICE) -> tuple[nn.Module, list[dict[str, Any]]]:

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.3,
                         total_iters=len(train_loader) * epochs)

    history = []
    best_state = None
    best_dev_f1 = -1
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader, 1):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            chunk_feat = batch["chunk_feat"].to(device)
            gs = batch["start"].to(device)
            ge = batch["end"].to(device)
            gi = batch["inside"].to(device)
            mask = batch["loss_mask"].float().to(device)

            ps, pe, pi, _ = model(ids, attn, chunk_feat)
            loss = token_loss(ps, gs, mask) + token_loss(pe, ge, mask) + token_loss(pi, gi, mask)

            loss.backward()

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        if dev_dataset is not None:
            dev_metrics = evaluate_exact(model, dev_dataset, (0.5, 0.5, 0.5), device=device)
            dev_f1 = dev_metrics["f1"]

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            history.append({"epoch": epoch, "loss": avg_loss, "dev_f1": dev_f1})
            print(f"Epoch {epoch} | loss {avg_loss:.4f} | dev F1 {dev_f1:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            history.append({"epoch": epoch, "loss": avg_loss})
            print(f"Epoch {epoch} | loss {avg_loss:.4f}")

    if dev_dataset is not None and best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def train_model_with_seed(seed: int, train_dataset: CoAMTokenDataset,
                          dev_dataset: Optional[CoAMTokenDataset] = None,
                          model_class: Type[nn.Module] = StartEndInsideModel,
                          model_name: str = MODEL_NAME, batch_size: int = BATCH_SIZE,
                          **kwargs) -> tuple[nn.Module, list[dict[str, Any]]]:
    set_seed(seed)
    model = model_class(model_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return train_model(model, train_loader, dev_dataset, **kwargs)
