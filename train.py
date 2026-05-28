"""
train.py — Шаг 3 (продолжение): Обучение моделей
═══════════════════════════════════════════════════════════════════════════════
Возможности:
  • Mixed Precision Training (AMP, float16) — ускорение на GPU
  • Cosine Annealing LR Scheduler с warmup
  • Early Stopping по метрике val_AUC
  • Gradient Clipping — стабильность градиентов LSTM
  • Label Smoothing в CrossEntropyLoss
  • Логирование loss/accuracy/AUC по эпохам
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as cfg
from src.models import build_cnn_bilstm, build_vit
from src.preprocessing import DeepfakeImageDataset, build_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Останавливает обучение, если метрика не улучшается patience эпох.
    Сохраняет лучшие веса модели.
    """
    def __init__(self, patience: int = cfg.EARLY_STOP, mode: str = "max"):
        self.patience  = patience
        self.mode      = mode
        self.best      = -np.inf if mode == "max" else np.inf
        self.counter   = 0
        self.triggered = False

    def step(self, metric: float) -> bool:
        """Returns True если нужно остановить обучение."""
        improved = (metric > self.best) if self.mode == "max" else (metric < self.best)
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


class WarmupCosineScheduler:
    """
    LR расписание: линейный warmup → cosine decay.
    
    Формула:
      epoch < warmup: lr = base_lr * epoch / warmup_epochs
      epoch >= warmup: lr = base_lr * 0.5 * (1 + cos(π * progress))
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.opt           = optimizer
        self.warmup        = warmup_epochs
        self.total         = total_epochs
        self.base_lr       = base_lr
        self.current_epoch = 0

    def step(self):
        e = self.current_epoch
        if e < self.warmup:
            lr = self.base_lr * (e + 1) / self.warmup
        else:
            progress = (e - self.warmup) / max(1, self.total - self.warmup)
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg in self.opt.param_groups:
            pg["lr"] = lr

        self.current_epoch += 1
        return lr


# ─────────────────────────────────────────────────────────────────────────────
# Одна эпоха обучения / валидации
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    mode:      str,                # "train" | "val" | "test"
    is_video:  bool = False,       # True → модель принимает (B, T, C, H, W)
) -> Dict[str, float]:
    """
    Прогоняет одну эпоху обучения или оценки.
    
    Returns:
        dict: {"loss": float, "acc": float, "auc": float}
    """
    is_train = (mode == "train")
    model.train(is_train)

    total_loss = 0.0
    all_labels: List[int]  = []
    all_probs:  List[float] = []
    correct = 0
    total   = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(cfg.DEVICE)
        labels = labels.to(cfg.DEVICE)

        with autocast(enabled=cfg.MIXED_PREC):
            if is_video:
                logits, _ = model(inputs)
            else:
                logits = model(inputs)

            loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # Gradient clipping (важно для LSTM — взрывные градиенты)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

        probs = torch.softmax(logits.detach(), dim=-1)[:, 1].cpu().numpy()
        preds = logits.detach().argmax(dim=-1).cpu()

        total_loss += loss.item() * labels.size(0)
        correct    += (preds == labels.cpu()).sum().item()
        total      += labels.size(0)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5   # только один класс в батче

    return {"loss": avg_loss, "acc": accuracy, "auc": auc}


# ─────────────────────────────────────────────────────────────────────────────
# Цикл обучения
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model_type: str,          # "vit" | "cnn_bilstm"
    data_dir:   str,
    save_dir:   str = cfg.CHECKPOINTS,
    num_epochs: int = cfg.NUM_EPOCHS,
) -> Dict:
    """
    Полный цикл обучения с валидацией.
    
    Args:
        model_type: тип модели
        data_dir:   директория с датасетом (split/class/images)
        save_dir:   куда сохранять чекпоинты
        num_epochs: максимальное число эпох
        
    Returns:
        history: словарь с метриками по эпохам
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Датасеты и загрузчики ──
    is_video = (model_type == "cnn_bilstm")

    if is_video:
        # Импортируем видео-датасет
        from src.preprocessing import DeepfakeVideoDataset
        DatasetClass = DeepfakeVideoDataset
        batch_size   = cfg.BATCH_SIZE_CNN
    else:
        DatasetClass = DeepfakeImageDataset
        batch_size   = cfg.BATCH_SIZE_VIT

    train_ds = DatasetClass(data_dir, split="train")
    val_ds   = DatasetClass(data_dir, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    log.info(f"Train: {len(train_ds)} / Val: {len(val_ds)} примеров")

    # ── Модель ──
    if model_type == "vit":
        model = build_vit()
        base_lr = cfg.LR_VIT
    else:
        model = build_cnn_bilstm()
        base_lr = cfg.LR_CNN

    # ── Функция потерь (Label Smoothing снижает переобучение) ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ── Оптимизатор (AdamW с weight decay для регуляризации) ──
    # Разные LR для backbone и новых слоёв (Discriminative LR)
    if model_type == "vit" and hasattr(model.vit, "blocks"):
        backbone_params = list(model.vit.blocks.parameters()) + \
                         list(model.vit.patch_embed.parameters() if hasattr(model.vit, "patch_embed") else [])
        head_params     = list(model.vit.head.parameters())
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": base_lr * 0.1},   # backbone обучается медленнее
            {"params": head_params,     "lr": base_lr},
        ], weight_decay=cfg.WEIGHT_DECAY)
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=cfg.WEIGHT_DECAY,
        )

    scheduler = WarmupCosineScheduler(optimizer, cfg.WARMUP_EPOCHS, num_epochs, base_lr)
    scaler    = GradScaler(enabled=cfg.MIXED_PREC and cfg.DEVICE == "cuda")
    stopper   = EarlyStopping(patience=cfg.EARLY_STOP, mode="max")

    history = {
        "train_loss": [], "train_acc": [], "train_auc": [],
        "val_loss":   [], "val_acc":   [], "val_auc":   [],
        "lr":         [],
    }
    best_auc  = 0.0
    best_path = os.path.join(save_dir, f"best_{model_type}.pth")

    log.info(f"\n{'='*55}")
    log.info(f"ОБУЧЕНИЕ МОДЕЛИ: {model_type.upper()}")
    log.info(f"{'='*55}")
    log.info(f"Устройство:  {cfg.DEVICE}")
    log.info(f"Эпохи:       {num_epochs}")
    log.info(f"Batch size:  {batch_size}")
    log.info(f"Base LR:     {base_lr}")
    log.info(f"{'='*55}")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        lr = scheduler.step()

        # Обучение
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            mode="train", is_video=is_video,
        )
        # Валидация
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, scaler,
            mode="val", is_video=is_video,
        )

        # Логирование
        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in val_metrics.items():
            history[f"val_{k}"].append(v)
        history["lr"].append(lr)

        log.info(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"LR: {lr:.2e} | "
            f"Train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} auc={train_metrics['auc']:.4f} | "
            f"Val:   loss={val_metrics['loss']:.4f}   acc={val_metrics['acc']:.4f}   auc={val_metrics['auc']:.4f}"
        )

        # Сохранение лучшей модели
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), best_path)
            log.info(f"  ✓ Лучшая модель сохранена (AUC={best_auc:.4f}): {best_path}")

        # Early stopping
        if stopper.step(val_metrics["auc"]):
            log.info(f"  Early stopping сработал на эпохе {epoch}")
            break

    elapsed = time.time() - start_time
    log.info(f"\nОбучение завершено за {elapsed/60:.1f} мин. Лучший AUC: {best_auc:.4f}")

    # Сохраняем историю обучения
    hist_path = os.path.join(save_dir, f"history_{model_type}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Визуализация кривых обучения
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: Dict, model_name: str, save_path: str = None):
    """
    Строит графики loss, accuracy, AUC, LR по эпохам.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Кривые обучения: {model_name}", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    metrics = [
        ("loss",  "Loss (Cross-Entropy)",  axes[0, 0]),
        ("acc",   "Accuracy",              axes[0, 1]),
        ("auc",   "AUC-ROC",               axes[1, 0]),
    ]

    for key, title, ax in metrics:
        ax.plot(epochs, history[f"train_{key}"], "b-o", markersize=4,
                label=f"Train", linewidth=2)
        ax.plot(epochs, history[f"val_{key}"],   "r-o", markersize=4,
                label=f"Val", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Эпоха")
        ax.legend()
        ax.grid(alpha=0.3)
        # Горизонтальная линия лучшего val
        best_val = max(history[f"val_{key}"]) if key != "loss" else min(history[f"val_{key}"])
        ax.axhline(best_val, color="r", linestyle="--", alpha=0.4,
                   label=f"Best={best_val:.4f}")

    # LR
    axes[1, 1].plot(epochs, history["lr"], "g-", linewidth=2)
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Эпоха")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save = save_path or os.path.join(cfg.PLOTS_DIR, f"03_training_{model_name}.png")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    log.info(f"Кривые обучения сохранены: {save}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection — Обучение")
    parser.add_argument("--model",      default="vit",
                        choices=["vit", "cnn_bilstm", "both"])
    parser.add_argument("--data_dir",   default=cfg.DATA_PROC)
    parser.add_argument("--epochs",     type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--save_dir",   default=cfg.CHECKPOINTS)
    args = parser.parse_args()

    models_to_train = ["vit", "cnn_bilstm"] if args.model == "both" else [args.model]

    for model_name in models_to_train:
        history = train_model(
            model_type=model_name,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            num_epochs=args.epochs,
        )
        plot_training_curves(history, model_name)


if __name__ == "__main__":
    main()
