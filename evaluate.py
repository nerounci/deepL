"""
evaluate.py — Шаг 4: Оценка работоспособности модели
═══════════════════════════════════════════════════════════════════════════════
Метрики:
  • AUC-ROC            — Стандартная метрика бинарной классификации
  • EER (Equal Error Rate) — Важна для биометрии / детекции мошенничества
  • AP (Average Precision)  — Площадь под PR-кривой
  • Confusion Matrix    — TP, TN, FP, FN с нормализацией
  • Calibration Curve   — Насколько вероятности модели откалиброваны
  • Threshold Analysis  — Как меняются TPR/FPR при смене порога

Визуализации:
  • ROC кривая
  • Precision-Recall кривая
  • Матрица ошибок
  • Calibration plot
  • Histogram предсказанных вероятностей
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as cfg
from src.models import build_vit, build_cnn_bilstm, build_ensemble
from src.preprocessing import DeepfakeImageDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Инференс
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(
    model:     torch.nn.Module,
    loader:    DataLoader,
    is_video:  bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Прогоняет модель на датасете и собирает предсказания.
    
    Returns:
        labels:    (N,) истинные метки
        probs:     (N,) вероятность класса 'fake' (P(fake))
        preds:     (N,) бинарные предсказания по порогу 0.5
    """
    model.eval()
    all_labels: List[int]   = []
    all_probs:  List[float] = []

    for inputs, labels in loader:
        inputs = inputs.to(cfg.DEVICE)

        if is_video:
            logits, _ = model(inputs)
        else:
            logits = model(inputs)

        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())

    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)
    preds_arr  = (probs_arr >= cfg.DECISION_THRESHOLD).astype(int)

    return labels_arr, probs_arr, preds_arr


# ─────────────────────────────────────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────────────────────────────────────
def compute_eer(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """
    Equal Error Rate — точка, где FPR == FNR.
    Чем ниже EER, тем лучше модель.
    
    Применяется в: биометрия, детекция мошенничества, дипфейки.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]
    return float(eer), float(eer_threshold)


def compute_all_metrics(
    labels: np.ndarray,
    probs:  np.ndarray,
    preds:  np.ndarray,
) -> Dict[str, float]:
    """
    Вычисляет полный набор метрик для бинарной классификации.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    eer, eer_thr = compute_eer(labels, probs)

    metrics = {
        "AUC-ROC":           float(roc_auc_score(labels, probs)),
        "AP (PR-AUC)":       float(average_precision_score(labels, probs)),
        "EER":               float(eer),
        "EER_threshold":     float(eer_thr),
        "Brier_Score":       float(brier_score_loss(labels, probs)),  # калиброванность
    }

    # Из confusion matrix
    if len(np.unique(labels)) == 2:
        cm = confusion_matrix(labels, preds)
        TN, FP, FN, TP = cm.ravel()
        N = TN + FP + FN + TP
        metrics.update({
            "Accuracy":        (TP + TN) / N,
            "Precision":       TP / (TP + FP + 1e-9),
            "Recall (TPR)":    TP / (TP + FN + 1e-9),
            "Specificity":     TN / (TN + FP + 1e-9),
            "F1":              2 * TP / (2 * TP + FP + FN + 1e-9),
            "FPR":             FP / (FP + TN + 1e-9),
            "TP":              int(TP), "TN": int(TN),
            "FP":              int(FP), "FN": int(FN),
        })

    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = ""):
    """Красивый вывод метрик в консоль."""
    log.info(f"\n{'='*55}")
    log.info(f"МЕТРИКИ ОЦЕНКИ: {model_name.upper()}")
    log.info(f"{'='*55}")
    key_metrics = [
        "AUC-ROC", "AP (PR-AUC)", "Accuracy", "F1",
        "Precision", "Recall (TPR)", "Specificity", "EER",
        "Brier_Score",
    ]
    for k in key_metrics:
        if k in metrics:
            bar = "█" * int(metrics[k] * 20)
            log.info(f"  {k:<18}: {metrics[k]:.4f}  {bar}")
    log.info(f"\n  Confusion Matrix: TP={metrics.get('TP',0)} TN={metrics.get('TN',0)} "
             f"FP={metrics.get('FP',0)} FN={metrics.get('FN',0)}")
    log.info(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────
# Визуализации
# ─────────────────────────────────────────────────────────────────────────────
def plot_evaluation(
    labels:     np.ndarray,
    probs:      np.ndarray,
    preds:      np.ndarray,
    metrics:    Dict[str, float],
    model_name: str = "",
    save_path:  str = None,
):
    """
    Комплексный дашборд оценки:
      [1] ROC-кривая          [2] PR-кривая
      [3] Confusion Matrix    [4] Calibration Plot
      [5] Histogram P(fake)   [6] Threshold Analysis
    """
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle(
        f"Оценка модели: {model_name}  |  AUC={metrics.get('AUC-ROC',0):.4f}  "
        f"F1={metrics.get('F1',0):.4f}",
        fontsize=14, fontweight="bold"
    )

    # ── 1. ROC-кривая ──
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, thresholds = roc_curve(labels, probs)
    eer, _ = compute_eer(labels, probs)
    auc = metrics.get("AUC-ROC", 0)

    ax1.plot(fpr, tpr, "b-", linewidth=2.5, label=f"ROC (AUC={auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax1.scatter([eer], [1 - eer], color="red", s=80, zorder=5, label=f"EER={eer:.4f}")
    ax1.fill_between(fpr, tpr, alpha=0.1, color="blue")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC-кривая")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ── 2. PR-кривая ──
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = metrics.get("AP (PR-AUC)", 0)
    baseline = labels.mean()

    ax2.plot(recall, precision, "g-", linewidth=2.5, label=f"PR (AP={ap:.4f})")
    ax2.axhline(baseline, color="k", linestyle="--", alpha=0.4,
                label=f"Baseline={baseline:.3f}")
    ax2.fill_between(recall, precision, alpha=0.1, color="green")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall кривая")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # ── 3. Матрица ошибок ──
    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Двойная аннотация: абс. значения + нормализованные
    annot = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i,j]}\n({cm_norm[i,j]:.1%})"

    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues",
        ax=ax3, cbar=True,
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 11},
    )
    ax3.set_xlabel("Предсказано")
    ax3.set_ylabel("Истинно")
    ax3.set_title("Матрица ошибок (нормализованная)")

    # ── 4. Calibration Plot ──
    ax4 = fig.add_subplot(gs[1, 0])
    fraction_pos, mean_pred = calibration_curve(labels, probs, n_bins=10)
    brier = metrics.get("Brier_Score", 0)

    ax4.plot(mean_pred, fraction_pos, "s-", color="orange", linewidth=2,
             label=f"Модель (Brier={brier:.4f})", markersize=7)
    ax4.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Идеальная калибровка")
    ax4.fill_between([0, 1], [0, 1], alpha=0.05, color="green")
    ax4.set_xlabel("Предсказанная вероятность")
    ax4.set_ylabel("Доля положительных (Fake)")
    ax4.set_title("Кривая калибровки")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # ── 5. Гистограмма P(fake) ──
    ax5 = fig.add_subplot(gs[1, 1])
    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]

    ax5.hist(real_probs, bins=30, alpha=0.7, color="#2196F3",
             label=f"Real (n={len(real_probs)})", density=True)
    ax5.hist(fake_probs, bins=30, alpha=0.7, color="#F44336",
             label=f"Fake (n={len(fake_probs)})", density=True)
    ax5.axvline(cfg.DECISION_THRESHOLD, color="black", linestyle="--",
                label=f"Порог={cfg.DECISION_THRESHOLD}")
    ax5.set_xlabel("P(Fake)")
    ax5.set_ylabel("Плотность")
    ax5.set_title("Распределение предсказанных вероятностей")
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    # ── 6. Анализ порогов ──
    ax6 = fig.add_subplot(gs[1, 2])
    thrs = np.linspace(0.1, 0.9, 50)
    f1s, recalls, precisions, specificities = [], [], [], []
    for thr in thrs:
        p = (probs >= thr).astype(int)
        cm_t = confusion_matrix(labels, p)
        if cm_t.shape == (2, 2):
            tn, fp, fn, tp = cm_t.ravel()
            f1s.append(2 * tp / (2 * tp + fp + fn + 1e-9))
            recalls.append(tp / (tp + fn + 1e-9))
            precisions.append(tp / (tp + fp + 1e-9))
            specificities.append(tn / (tn + fp + 1e-9))
        else:
            f1s.append(0); recalls.append(0); precisions.append(0); specificities.append(0)

    ax6.plot(thrs, f1s,        "b-", linewidth=2, label="F1")
    ax6.plot(thrs, recalls,    "g-", linewidth=2, label="Recall (TPR)")
    ax6.plot(thrs, precisions, "r-", linewidth=2, label="Precision")
    ax6.axvline(cfg.DECISION_THRESHOLD, color="black", linestyle="--", alpha=0.5)
    ax6.set_xlabel("Порог классификации")
    ax6.set_ylabel("Значение метрики")
    ax6.set_title("Метрики vs Порог")
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    save = save_path or os.path.join(cfg.PLOTS_DIR, f"04_evaluation_{model_name}.png")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    log.info(f"Графики оценки сохранены: {save}")
    plt.close()
    return save


# ─────────────────────────────────────────────────────────────────────────────
# Сравнение моделей
# ─────────────────────────────────────────────────────────────────────────────
def compare_models(results: Dict[str, Dict], save_path: str = None):
    """
    Сравнительный bar chart всех моделей по ключевым метрикам.
    """
    key_metrics = ["AUC-ROC", "F1", "Accuracy", "AP (PR-AUC)"]
    model_names = list(results.keys())

    fig, axes = plt.subplots(1, len(key_metrics), figsize=(16, 5))
    fig.suptitle("Сравнение моделей", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(model_names)]

    for i, metric in enumerate(key_metrics):
        values = [results[m].get(metric, 0) for m in model_names]
        bars = axes[i].bar(model_names, values, color=colors, alpha=0.85, edgecolor="white")
        axes[i].set_title(metric, fontsize=12)
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

    plt.tight_layout()
    save = save_path or os.path.join(cfg.PLOTS_DIR, "05_model_comparison.png")
    plt.savefig(save, dpi=150, bbox_inches="tight")
    log.info(f"Сравнение моделей сохранено: {save}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Демо-оценка на синтетических данных
# ─────────────────────────────────────────────────────────────────────────────
def demo_evaluation_with_synthetic_predictions(n_samples: int = 200):
    """
    Демонстрация метрик на синтетических предсказаниях.
    Используется когда модель ещё не обучена.
    """
    np.random.seed(cfg.SEED)

    # Симулируем предсказания хорошей модели (AUC ≈ 0.92)
    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Real: P(fake) ≈ 0.15 (низкая вероятность)
    # Fake: P(fake) ≈ 0.82 (высокая вероятность)
    real_preds = np.random.beta(2, 10, n_samples // 2)      # сосредоточены у 0
    fake_preds = np.random.beta(10, 2, n_samples // 2)      # сосредоточены у 1
    probs = np.concatenate([real_preds, fake_preds])
    probs = np.clip(probs, 0.01, 0.99)
    preds = (probs >= cfg.DECISION_THRESHOLD).astype(int)

    metrics = compute_all_metrics(labels, probs, preds)
    print_metrics(metrics, "Демо (синтетические предсказания)")
    save = plot_evaluation(labels, probs, preds, metrics,
                           model_name="Demo Synthetic")
    return metrics, save


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection — Оценка")
    parser.add_argument("--checkpoint",  default=None,
                        help="Путь к чекпоинту модели (.pth)")
    parser.add_argument("--model",       default="vit",
                        choices=["vit", "cnn_bilstm"])
    parser.add_argument("--data_dir",    default=cfg.DATA_PROC)
    parser.add_argument("--demo",        action="store_true",
                        help="Демо-оценка на синтетических данных")
    args = parser.parse_args()

    if args.demo or not args.checkpoint:
        log.info("Запуск демо-оценки на синтетических данных...")
        metrics, plot = demo_evaluation_with_synthetic_predictions()
        log.info(f"\n✅ Демо-оценка завершена. График: {plot}")
        return

    # Реальная оценка
    if args.model == "vit":
        model = build_vit()
    else:
        model = build_cnn_bilstm()

    model.load_state_dict(torch.load(args.checkpoint, map_location=cfg.DEVICE))

    test_ds     = DeepfakeImageDataset(args.data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=cfg.NUM_WORKERS)

    labels, probs, preds = predict(model, test_loader, is_video=(args.model == "cnn_bilstm"))
    metrics = compute_all_metrics(labels, probs, preds)
    print_metrics(metrics, args.model)
    plot_evaluation(labels, probs, preds, metrics, args.model)

    # Сохраняем метрики
    out_path = os.path.join(cfg.OUTPUTS, f"metrics_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
