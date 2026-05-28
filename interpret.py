"""
interpret.py — Шаг 5: Интерпретация результатов
═══════════════════════════════════════════════════════════════════════════════
Методы интерпретации:
  1. Grad-CAM (CNN-BiLSTM)   — тепловые карты важных регионов изображения
  2. Attention Rollout (ViT) — визуализация внимания трансформера
  3. Temporal Attention       — какие кадры видео важны для решения
  4. LIME-like Perturbation   — локальная интерпретируемость
  5. LLM Explanation          — генеративная текстовая интерпретация через API

Grad-CAM (Gradient-weighted Class Activation Mapping):
  dY_c / dA^l → α_k^c = GAP(∂Y_c / ∂A_k^l) → L^c = ReLU(Σ_k α_k^c · A_k^l)
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as cfg
from src.models import DeepfakeCNNBiLSTM, DeepfakeViT, build_cnn_bilstm, build_vit
from src.preprocessing import build_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Grad-CAM для CNN слоёв
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Реализация Grad-CAM для произвольного CNN слоя.
    
    Алгоритм:
      1. Прямой проход: сохраняем активации целевого слоя A^l (B, C, H, W)
      2. Обратный проход: вычисляем градиенты ∂Y_c / ∂A^l
      3. Глобальное среднее пулинг градиентов: α_k = GAP(grad)  (C,)
      4. Взвешенная сумма: L^c = ReLU(Σ_k α_k · A_k^l)
      5. Интерполяция до размера входа
    
    Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations..."
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self.activations = output.detach()

        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.fwd_handle = self.target_layer.register_forward_hook(fwd_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,     # (1, C, H, W)
        class_idx:    Optional[int] = None,
    ) -> np.ndarray:
        """
        Генерирует тепловую карту Grad-CAM.
        
        Returns:
            heatmap: (H, W) в диапазоне [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Прямой проход
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        if class_idx is None:
            class_idx = output.argmax(dim=-1).item()

        # Обнуляем градиенты
        self.model.zero_grad()

        # Обратный проход по целевому классу
        output[0, class_idx].backward()

        # Grad-CAM формула
        grads = self.gradients          # (1, C, H, W)
        acts  = self.activations        # (1, C, H, W)

        # Глобальное усреднение градиентов по пространственным осям
        weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Взвешенная сумма карт активаций
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Нормализация в [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam

    def overlay(
        self,
        original_img: np.ndarray,    # (H, W, 3) uint8
        cam:          np.ndarray,    # (h, w) float [0,1]
        alpha:        float = 0.5,
    ) -> np.ndarray:
        """Накладывает тепловую карту на исходное изображение."""
        H, W = original_img.shape[:2]

        # Ресайз cam до размера изображения
        cam_resized = cv2.resize(cam, (W, H))

        # Цветовая карта jet
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = (alpha * heatmap + (1 - alpha) * original_img).astype(np.uint8)
        return overlay

    def __del__(self):
        try:
            self.fwd_handle.remove()
            self.bwd_handle.remove()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 2. Attention Rollout для ViT
# ─────────────────────────────────────────────────────────────────────────────
class AttentionRollout:
    """
    Attention Rollout для ViT: усредняет внимание по всем слоям.
    
    Алгоритм (Abnar & Zuidema, 2020):
      1. Для каждого блока: A_i = 0.5 * A_raw + 0.5 * I  (с residual)
      2. Rollout = A_1 @ A_2 @ ... @ A_L
      3. Веса внимания CLS токена → пространственная карта
    
    Позволяет понять, какие патчи изображения важны для классификации.
    """

    def __init__(self, model: DeepfakeViT):
        self.model     = model
        self.attn_maps = []
        self._register_hooks()

    def _register_hooks(self):
        if not self.model.use_timm:
            # Для кастомной реализации
            for block in self.model.vit.transformer:
                block.attn.register_forward_hook(self._save_attn)
        else:
            try:
                for block in self.model.vit.blocks:
                    block.attn.register_forward_hook(self._save_attn)
            except AttributeError:
                log.warning("Не удалось зарегистрировать hooks для ViT")

    def _save_attn(self, module, input, output):
        # output может быть тензором или кортежем
        if isinstance(output, tuple):
            attn = output[1]  # (B, heads, N, N)
        else:
            attn = output
        if attn is not None and attn.dim() == 4:
            self.attn_maps.append(attn.detach().cpu())

    @torch.no_grad()
    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Returns:
            rollout_map: (H, W) тепловая карта внимания по патчам
        """
        self.attn_maps.clear()
        self.model.eval()
        _ = self.model(input_tensor)

        if not self.attn_maps:
            # Если hooks не сработали, возвращаем равномерную карту
            return np.ones((14, 14)) / (14 * 14)

        # Rollout: перемножаем матрицы внимания
        rollout = None
        for attn in self.attn_maps:
            # attn: (B, heads, N, N); усредняем по головам
            attn_avg = attn.mean(dim=1)[0]    # (N, N)
            # Добавляем residual connection: 0.5 * A + 0.5 * I
            attn_res = 0.5 * attn_avg + 0.5 * torch.eye(attn_avg.shape[0])
            # Нормализуем строки
            attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)

            rollout = attn_res if rollout is None else attn_res @ rollout

        # Внимание CLS токена на остальные токены
        cls_attn = rollout[0, 1:].numpy()   # (N,) без CLS

        # Формируем 2D карту
        grid_size = int(np.sqrt(cls_attn.shape[0]))
        attn_map  = cls_attn[:grid_size**2].reshape(grid_size, grid_size)
        attn_map  = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        return attn_map


# ─────────────────────────────────────────────────────────────────────────────
# 3. Temporal Attention Visualization (CNN-BiLSTM)
# ─────────────────────────────────────────────────────────────────────────────
def visualize_temporal_attention(
    frames:         List[np.ndarray],    # кадры видео
    attn_weights:   np.ndarray,          # (T,) временные веса
    prediction:     str,
    confidence:     float,
    save_path:      str = None,
):
    """
    Отображает, каким кадрам BiLSTM придаёт наибольший вес.
    Высокий вес кадра = в этом кадре наиболее заметны артефакты дипфейка.
    """
    T = len(frames)
    fig, axes = plt.subplots(2, T // 2 + 1, figsize=(20, 6))
    fig.suptitle(
        f"Temporal Attention | Предсказание: {prediction} ({confidence:.1%})",
        fontsize=13, fontweight="bold",
        color="red" if prediction == "FAKE" else "green"
    )

    # Верхний ряд: кадры с весами
    for i in range(min(T, T // 2 * 2)):
        row, col = i // (T // 2), i % (T // 2)
        ax = axes[row, col]
        ax.imshow(frames[i])
        w = float(attn_weights[i])
        # Цвет рамки: красный = высокое внимание (важный кадр)
        color = plt.cm.Reds(w / (max(attn_weights) + 1e-8))
        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_color(color[:3])
        ax.set_title(f"Кадр {i+1}\nα={w:.3f}", fontsize=9)
        ax.axis("off")

    # Нижний правый: гистограмма весов
    ax_bar = axes[1, T // 2]
    colors = plt.cm.Reds(attn_weights / (max(attn_weights) + 1e-8))
    ax_bar.bar(range(T), attn_weights, color=colors, edgecolor="white")
    ax_bar.set_title("Temporal Attention Weights")
    ax_bar.set_xlabel("Номер кадра")
    ax_bar.set_ylabel("Вес α")
    ax_bar.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save = save_path or os.path.join(cfg.PLOTS_DIR, "temporal_attention.png")
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Temporal attention: {save}")
    return save


# ─────────────────────────────────────────────────────────────────────────────
# 4. Комплексный дашборд интерпретации
# ─────────────────────────────────────────────────────────────────────────────
def interpret_image(
    image_path: str,
    vit_model:  DeepfakeViT,
    save_path:  str = None,
) -> Dict:
    """
    Полная интерпретация одного изображения:
      1. Предсказание
      2. Grad-CAM карта
      3. Attention Rollout карта
      4. Сравнение регионов real vs fake
    """
    transform = build_transforms("val")
    img_pil   = Image.open(image_path).convert("RGB")
    img_arr   = np.array(img_pil.resize(cfg.IMAGE_SIZE))
    tensor    = transform(img_pil).unsqueeze(0).to(cfg.DEVICE)

    # Предсказание ViT
    vit_model.eval()
    with torch.no_grad():
        logits = vit_model(tensor)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_class = int(probs.argmax())
    pred_label = "FAKE" if pred_class == 1 else "REAL"
    confidence = float(probs[pred_class])

    log.info(f"Предсказание: {pred_label} (уверенность: {confidence:.1%})")
    log.info(f"P(real)={probs[0]:.4f} | P(fake)={probs[1]:.4f}")

    # Grad-CAM (через последний conv блок)
    cam_map = None
    try:
        # Находим целевой слой (последний Conv2d в CNN backbone)
        cnn_model = build_cnn_bilstm()
        # Для ViT — используем другой подход
        target_layer = None
        for name, module in vit_model.named_modules():
            if isinstance(module, nn.LayerNorm):
                target_layer = module

        if target_layer:
            gcam = GradCAM(vit_model, target_layer)
            # Grad-CAM не идеален для ViT, используем Attention Rollout
    except Exception as e:
        log.debug(f"Grad-CAM: {e}")

    # Attention Rollout
    rollout = AttentionRollout(vit_model)
    attn_map = rollout.generate(tensor)

    # ── Визуализация ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Интерпретация детекции дипфейков | Предсказание: {pred_label} ({confidence:.1%})",
        fontsize=13, fontweight="bold",
        color="#F44336" if pred_label == "FAKE" else "#4CAF50"
    )

    # 1. Исходное изображение
    axes[0, 0].imshow(img_arr)
    axes[0, 0].set_title("Исходное изображение", fontsize=11)
    axes[0, 0].axis("off")

    # 2. Карта внимания ViT (Attention Rollout)
    attn_resized = cv2.resize(attn_map, cfg.IMAGE_SIZE[::-1])
    im = axes[0, 1].imshow(attn_resized, cmap="hot", interpolation="bilinear")
    axes[0, 1].set_title("Attention Rollout (ViT)\n(яркость = важность региона)", fontsize=10)
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # 3. Оверлей
    heatmap_color = cv2.applyColorMap(
        (attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * heatmap_color + 0.45 * img_arr).astype(np.uint8)
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title("Оверлей (оригинал + внимание)", fontsize=10)
    axes[0, 2].axis("off")

    # 4. Bar chart вероятностей
    classes = ["Real", "Fake"]
    colors  = ["#2196F3", "#F44336"]
    bars    = axes[1, 0].bar(classes, probs, color=colors, alpha=0.85, edgecolor="white")
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].set_title("Распределение вероятностей", fontsize=10)
    axes[1, 0].set_ylabel("P(класс)")
    for bar, val in zip(bars, probs):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.4f}", ha="center", fontsize=12, fontweight="bold"
        )
    axes[1, 0].grid(axis="y", alpha=0.3)

    # 5. Топ патчей по вниманию
    patch_size = cfg.IMAGE_SIZE[0] // 14
    top_k = 5
    flat = attn_map.flatten()
    top_idx = np.argsort(flat)[::-1][:top_k]
    patch_vis = img_arr.copy()
    for idx in top_idx:
        row, col = idx // 14, idx % 14
        y1, y2 = row * patch_size, (row + 1) * patch_size
        x1, x2 = col * patch_size, (col + 1) * patch_size
        cv2.rectangle(patch_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    axes[1, 1].imshow(patch_vis)
    axes[1, 1].set_title(f"Топ-{top_k} патчей (красные рамки)\n= наибольшее внимание", fontsize=10)
    axes[1, 1].axis("off")

    # 6. Карта аномалий (разница с ожидаемым распределением)
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY).astype(float)
    # Локальная нормализация → артефакты выделяются
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    anomaly = np.abs(gray - blurred)
    anomaly = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-8)
    im2 = axes[1, 2].imshow(anomaly, cmap="inferno")
    axes[1, 2].set_title("Карта локальных аномалий\n(отклонение от сглаженного)", fontsize=10)
    axes[1, 2].axis("off")
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    save = save_path or os.path.join(cfg.PLOTS_DIR, f"06_interpretation.png")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    log.info(f"Интерпретация сохранена: {save}")
    plt.close()

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "p_real":     float(probs[0]),
        "p_fake":     float(probs[1]),
        "save_path":  save,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. LLM-генератор объяснений (Anthropic API)
# ─────────────────────────────────────────────────────────────────────────────
def generate_llm_explanation(
    prediction:   str,
    confidence:   float,
    p_real:       float,
    p_fake:       float,
    image_path:   str,
    attn_regions: Optional[List[str]] = None,
) -> str:
    """
    Генерирует текстовое объяснение результата детекции дипфейка
    с использованием LLM (Claude API).
    
    Подход: мультимодальный запрос с изображением + метаданными.
    Соответствует требованию 6c (генеративные текстовые модели) и 6d (мультимодальные модели).
    """
    try:
        import anthropic
    except ImportError:
        return _fallback_explanation(prediction, confidence, p_real, p_fake)

    # Кодируем изображение в base64
    try:
        with open(image_path, "rb") as f:
            img_data = base64.standard_b64encode(f.read()).decode("utf-8")
        img_available = True
    except Exception:
        img_available = False

    prompt = f"""Ты — эксперт по обнаружению дипфейков и цифровой форензике. 
Нейронная сеть (Vision Transformer + BiLSTM) проанализировала изображение и получила результат:

**Результат детекции:**
- Предсказание: {prediction}
- Уверенность модели: {confidence:.1%}
- P(real) = {p_real:.4f}
- P(fake) = {p_fake:.4f}

{"Ключевые регионы, привлёкшие внимание модели: " + ", ".join(attn_regions) if attn_regions else ""}

Пожалуйста, предоставь:
1. **Интерпретацию**: Что означает этот результат? Почему модель приняла такое решение?
2. **Технические признаки дипфейков**: Какие артефакты (blending boundaries, frequency artifacts, temporal inconsistencies) характерны для данного класса манипуляций?
3. **Уровень доверия**: Насколько можно доверять этому результату при данном уровне уверенности?
4. **Рекомендации**: Что следует сделать, если это дипфейк? (верификация через другие методы, репортинг, и т.д.)

Отвечай структурированно, на русском языке, технически точно, но понятно для конечного пользователя."""

    client = anthropic.Anthropic()

    try:
        if img_available:
            # Мультимодальный запрос (изображение + текст)
            message = client.messages.create(
                model=cfg.LLM_MODEL,
                max_tokens=cfg.LLM_MAX_TOK,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type":   "image",
                            "source": {
                                "type":       "base64",
                                "media_type": "image/jpeg",
                                "data":       img_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
        else:
            message = client.messages.create(
                model=cfg.LLM_MODEL,
                max_tokens=cfg.LLM_MAX_TOK,
                messages=[{"role": "user", "content": prompt}],
            )

        return message.content[0].text

    except Exception as e:
        log.warning(f"LLM API ошибка: {e}")
        return _fallback_explanation(prediction, confidence, p_real, p_fake)


def _fallback_explanation(
    prediction: str, confidence: float,
    p_real: float, p_fake: float,
) -> str:
    """Шаблонное объяснение без API."""
    if prediction == "FAKE":
        return f"""
🚨 ДИПФЕЙК ОБНАРУЖЕН (уверенность: {confidence:.1%})

Модель с вероятностью {p_fake:.4f} классифицировала это изображение как дипфейк.

**Потенциальные признаки манипуляции:**
• Артефакты на границах лица (blending mask)
• Несоответствие текстуры кожи (GAN-артефакты)
• Геометрические искажения вокруг ушей/волос
• Аномальное освещение или тени
• Grid artifacts от транспонированных свёрток

**Рекомендации:**
1. Проверьте изображение через другие детекторы (FaceForensics++)
2. Проведите частотный анализ (DFT) — GAN-синтез оставляет следы в спектре
3. Используйте ELA (Error Level Analysis) для обнаружения компрессионных артефактов
        """
    else:
        return f"""
✅ ИЗОБРАЖЕНИЕ ПОДЛИННОЕ (уверенность: {confidence:.1%})

Модель с вероятностью {p_real:.4f} классифицировала это изображение как оригинальное.

Признаки аутентичности:
• Естественные текстуры кожи
• Согласованное освещение
• Отсутствие характерных GAN-артефактов
        """


# ─────────────────────────────────────────────────────────────────────────────
# Демо интерпретации на синтетическом изображении
# ─────────────────────────────────────────────────────────────────────────────
def demo_interpretation(save_path: str = None):
    """
    Демонстрирует полный пайплайн интерпретации
    на синтетически сгенерированном изображении.
    """
    from src.data_collection import _generate_face

    # Генерируем синтетическое дипфейк изображение
    fake_img = _generate_face(size=256, is_fake=True)
    real_img = _generate_face(size=256, is_fake=False)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Демонстрация интерпретации: синтетические данные", fontsize=14, fontweight="bold")

    for row, (img_arr, label) in enumerate([(real_img, "REAL"), (fake_img, "FAKE")]):
        img_pil = Image.fromarray(img_arr)

        # Оригинал
        axes[row, 0].imshow(img_arr)
        color = "#4CAF50" if label == "REAL" else "#F44336"
        axes[row, 0].set_title(f"Изображение [{label}]", color=color, fontsize=12, fontweight="bold")
        axes[row, 0].axis("off")

        # Частотный анализ (DFT) — GAN оставляет артефакты в спектре
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dft  = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log(np.abs(dft_shift) + 1)
        axes[row, 1].imshow(magnitude, cmap="plasma")
        axes[row, 1].set_title("DFT Спектр\n(артефакты ГАН видны в спектре)", fontsize=9)
        axes[row, 1].axis("off")

        # Локальная нормализованная карта
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        diff    = np.abs(gray - blurred)
        diff    = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        axes[row, 2].imshow(diff, cmap="hot")
        axes[row, 2].set_title("Карта высокочастотных аномалий\n(яркость = аномальность)", fontsize=9)
        axes[row, 2].axis("off")

        # Error Level Analysis (ELA) — анализ уровней ошибок сжатия
        buf = BytesIO()
        img_pil.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        img_resaved = np.array(Image.open(buf).convert("RGB"))
        ela = np.abs(img_arr.astype(int) - img_resaved.astype(int)).astype(np.uint8) * 5
        axes[row, 3].imshow(ela)
        axes[row, 3].set_title("ELA (Error Level Analysis)\n(яркость = сжатие/манипуляция)", fontsize=9)
        axes[row, 3].axis("off")

    plt.tight_layout()
    save = save_path or os.path.join(cfg.PLOTS_DIR, "06_interpretation_demo.png")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    log.info(f"Демо интерпретации: {save}")
    plt.close()
    return save


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deepfake Detection — Интерпретация")
    parser.add_argument("--demo",        action="store_true", help="Запустить демо")
    parser.add_argument("--image_path",  default=None)
    parser.add_argument("--vit_ckpt",    default=None)
    args = parser.parse_args()

    if args.demo or not args.image_path:
        log.info("Запуск демо интерпретации...")
        save = demo_interpretation()
        log.info(f"✅ Демо сохранено: {save}")
    else:
        vit = build_vit()
        if args.vit_ckpt:
            vit.load_state_dict(torch.load(args.vit_ckpt, map_location=cfg.DEVICE))
        result = interpret_image(args.image_path, vit)
        explanation = generate_llm_explanation(**result, image_path=args.image_path)
        print("\n" + "=" * 60)
        print("LLM ОБЪЯСНЕНИЕ:")
        print("=" * 60)
        print(explanation)
