"""
models.py — Шаг 3: Архитектуры нейронных сетей
═══════════════════════════════════════════════════════════════════════════════
Модель 1: EfficientNet-B0 + BiLSTM + Temporal Attention
  ├── CNN Backbone (EfficientNet-B0): извлечение признаков из кадра
  ├── BiLSTM: моделирование временных зависимостей между кадрами
  └── Temporal Attention: взвешивание информативных кадров

Модель 2: Vision Transformer (ViT-B/16) fine-tuned
  ├── Patch Embedding: 16×16 патчи → токены
  ├── 12× Transformer Encoder с Multi-Head Self-Attention
  └── CLS токен → классификатор

Ensemble: взвешенная комбинация обеих моделей
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import math
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as cfg

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные модули
# ─────────────────────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """
    Мягкое внимание (soft attention) по временной оси.
    
    Идея: не все кадры видео одинаково информативны.
    Кадры с явными артефактами дипфейка должны получать больший вес.
    
    Формула:
        score_t = v^T · tanh(W · h_t + b)
        alpha_t = softmax(score_t)  
        context = Σ alpha_t · h_t
    """

    def __init__(self, hidden_dim: int, attention_dim: int = cfg.ATTENTION_DIM):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)
            weights: (batch, seq_len) — интерпретируемые веса кадров
        """
        energy   = torch.tanh(self.W(lstm_out))      # (B, T, attention_dim)
        scores   = self.v(energy).squeeze(-1)          # (B, T)
        weights  = F.softmax(scores, dim=-1)           # (B, T) — сумма = 1
        context  = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden)
        return context, weights


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation блок для калибровки каналов CNN.
    Помогает модели фокусироваться на наиболее релевантных картах признаков.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ─────────────────────────────────────────────────────────────────────────────
# Модель 1: CNN Backbone + BiLSTM + Attention
# ─────────────────────────────────────────────────────────────────────────────
class DeepfakeCNNBiLSTM(nn.Module):
    """
    Детектор дипфейков для видеопоследовательностей.
    
    Архитектурный поток:
    [B, T, 3, 224, 224]
          │
          ▼ (reshape: B*T примеров)
    EfficientNet-B0                ← ImageNet pretrained
    [B*T, 1280, 1, 1]
          │ flatten
    [B*T, 1280]
          │ SE-block адаптация
          │ reshape → [B, T, 1280]
          ▼
    BiLSTM (256 hidden, 2 layers)
    [B, T, 512]      ← 256*2 (двунаправленный)
          │
    Temporal Attention
    [B, 512]         ← взвешенный контекст
          │
    FC: 512→256→2    ← dropout 0.3
    """

    def __init__(self):
        super().__init__()

        # ── CNN backbone ──
        effnet = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Удаляем финальный AdaptiveAvgPool и классификатор
        self.cnn = nn.Sequential(*list(effnet.children())[:-1])   # [B*T, 1280, 1, 1]
        self.feature_dim = 1280

        # Дополнительный SE-block поверх EfficientNet
        self.se = SEBlock(self.feature_dim)

        # ── BiLSTM ──
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,                 # h_t = [h_fwd; h_bwd]
            dropout=cfg.LSTM_DROPOUT if cfg.LSTM_LAYERS > 1 else 0.0,
        )
        lstm_out_dim = cfg.LSTM_HIDDEN * 2      # двунаправленный

        # ── Temporal Attention ──
        self.attention = TemporalAttention(lstm_out_dim)

        # ── Классификатор ──
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, cfg.NUM_CLASSES),
        )

        self._init_weights()

    def _init_weights(self):
        """Инициализация Xavier для линейных слоёв."""
        for m in [self.classifier, self.attention.W, self.attention.v]:
            for layer in (m.modules() if hasattr(m, "modules") else [m]):
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,           # (B, T, C, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = x.shape

        # ── Шаг 1: Извлечение признаков CNN ──
        x_flat = x.view(B * T, C, H, W)
        feat   = self.cnn(x_flat)          # (B*T, 1280, 1, 1)
        feat   = self.se(feat)             # SE-калибровка
        feat   = feat.view(B, T, -1)       # (B, T, 1280)

        # ── Шаг 2: BiLSTM ──
        lstm_out, (h_n, c_n) = self.lstm(feat)   # (B, T, 512)

        # ── Шаг 3: Temporal Attention ──
        context, attn_weights = self.attention(lstm_out)   # (B, 512), (B, T)

        # ── Шаг 4: Классификация ──
        logits = self.classifier(context)   # (B, 2)

        return logits, attn_weights

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает карты признаков последнего CNN блока (для Grad-CAM)."""
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        # Извлекаем карты до глобального пулинга
        features = list(self.cnn.children())
        intermediate = nn.Sequential(*features[:-1])  # до AdaptiveAvgPool
        return intermediate(x_flat)


# ─────────────────────────────────────────────────────────────────────────────
# Модель 2: Vision Transformer (ViT-B/16)
# ─────────────────────────────────────────────────────────────────────────────
class DeepfakeViT(nn.Module):
    """
    Vision Transformer для покадровой детекции дипфейков.
    
    Ключевые компоненты ViT:
    ┌─────────────────────────────────────────────────────┐
    │ Image 224×224 → 196 патчей (14×14 сетка, 16×16 пикс) │
    │ + 1 CLS токен → 197 токенов                          │
    │                                                       │
    │  12× Transformer Encoder:                            │
    │    LayerNorm → Multi-Head Self-Attention (12 heads)  │
    │    Residual  → LayerNorm → FFN (3072 нейрона)        │
    │    Residual                                           │
    │                                                       │
    │ CLS[0] → Linear(768 → 2) → [P_real, P_fake]         │
    └─────────────────────────────────────────────────────┘
    
    Self-Attention позволяет каждому патчу взаимодействовать
    со всеми другими патчами — глобальный контекст без индуктивного
    смещения свёрток. Это критично для обнаружения дипфейков:
    артефакты на одном участке лица взаимодействуют с другими.
    """

    def __init__(self):
        super().__init__()

        try:
            import timm
            self.vit = timm.create_model(
                cfg.VIT_MODEL_NAME,
                pretrained=cfg.VIT_PRETRAINED,
                num_classes=cfg.NUM_CLASSES,
                drop_rate=cfg.VIT_DROP_RATE,
                attn_drop_rate=cfg.VIT_ATTN_DROP_RATE,
            )
            self.use_timm = True
            log.info(f"ViT загружен через timm: {cfg.VIT_MODEL_NAME}")
        except ImportError:
            log.warning("timm не установлен. Используем кастомную реализацию ViT.")
            self.vit = self._build_custom_vit()
            self.use_timm = False

    def _build_custom_vit(self) -> nn.Module:
        """
        Упрощённая кастомная реализация ViT для случая,
        когда timm недоступен.
        """
        return _SimpleViT(
            image_size=cfg.IMAGE_SIZE[0],
            patch_size=16,
            num_classes=cfg.NUM_CLASSES,
            dim=768,
            depth=6,        # 6 вместо 12 для скорости
            heads=8,
            mlp_dim=2048,
            dropout=cfg.VIT_DROP_RATE,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — изображение лица
        Returns:
            logits: (B, 2)
        """
        return self.vit(x)

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Извлекает карты внимания последнего трансформер-блока.
        Используется для визуализации, какие патчи важны для решения.
        
        Returns:
            attention: (B, heads, tokens, tokens)
        """
        if not self.use_timm:
            return None

        attn_maps = []

        def hook_fn(module, input, output):
            attn_maps.append(output)

        # Регистрируем hook на последний attention блок
        last_block = list(self.vit.blocks)[-1]
        handle = last_block.attn.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.vit(x)

        handle.remove()
        return attn_maps[0] if attn_maps else None


class _SimpleViT(nn.Module):
    """
    Упрощённая реализация Vision Transformer без timm.
    Полностью воспроизводит логику оригинальной статьи:
    'An Image is Worth 16x16 Words' (Dosovitskiy et al., 2020)
    """

    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "image_size должен делиться на patch_size"

        num_patches = (image_size // patch_size) ** 2
        patch_dim   = 3 * patch_size * patch_size

        # Patch embedding: линейная проекция каждого патча
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.patch_size = patch_size

        # Learnable CLS токен + позиционные эмбеддинги
        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout       = nn.Dropout(dropout)

        # Transformer encoder блоки
        self.transformer = nn.Sequential(
            *[_TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Разбивает изображение (B, C, H, W) на патчи (B, N, patch_dim)."""
        B, C, H, W = x.shape
        p = self.patch_size
        # Unfold: H → H//p блоков, W → W//p блоков
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)   # (B, H//p, W//p, C, p, p)
        x = x.reshape(B, -1, C * p * p)    # (B, N, patch_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Патчи
        patches = self._to_patches(x)                         # (B, N, patch_dim)
        tokens  = self.patch_embed(patches)                    # (B, N, dim)

        # Prepend CLS токен
        cls     = self.cls_token.expand(B, -1, -1)            # (B, 1, dim)
        tokens  = torch.cat([cls, tokens], dim=1)             # (B, N+1, dim)
        tokens  = tokens + self.pos_embedding                  # добавляем pos. embed
        tokens  = self.dropout(tokens)

        # Transformer
        tokens  = self.transformer(tokens)                     # (B, N+1, dim)
        tokens  = self.norm(tokens)

        # CLS токен → классификация
        return self.head(tokens[:, 0])                        # (B, num_classes)


class _MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA).
    
    Для каждой головы:
        Q = XW_Q,  K = XW_K,  V = XW_V
        Attention(Q, K, V) = softmax(QK^T / √d_k) · V
    
    Все головы конкатенируются и проецируются обратно:
        MHSA(X) = Concat(head_1, ..., head_h) · W_O
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads    = heads
        self.d_k      = dim // heads
        self.scale    = math.sqrt(self.d_k)

        self.qkv      = nn.Linear(dim, dim * 3, bias=False)
        self.proj_out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = self.heads

        qkv = self.qkv(x).reshape(B, N, 3, H, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, N, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        attn = (Q @ K.transpose(-2, -1)) / self.scale   # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V)                               # (B, H, N, d_k)
        out = out.transpose(1, 2).reshape(B, N, D)     # (B, N, D)
        return self.proj_out(out)


class _TransformerBlock(nn.Module):
    """Pre-LN Transformer блок: LayerNorm → Attention/FFN → Residual."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.attn   = _MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2  = nn.LayerNorm(dim)
        self.ffn    = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # Self-Attention + residual
        x = x + self.ffn(self.norm2(x))    # FFN + residual
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────
class DeepfakeEnsemble(nn.Module):
    """
    Взвешенный ансамбль CNN-BiLSTM и ViT.
    
    Финальная вероятность:
        P_fake = w_cnn * P_fake_cnn + w_vit * P_fake_vit
    
    Где w_cnn + w_vit = 1.
    
    CNN-BiLSTM лучше обнаруживает:  темпоральные несоответствия
    ViT лучше обнаруживает:         глобальные пространственные артефакты
    """

    def __init__(self, cnn_model: DeepfakeCNNBiLSTM, vit_model: DeepfakeViT):
        super().__init__()
        self.cnn_model = cnn_model
        self.vit_model = vit_model
        self.w_cnn = cfg.ENSEMBLE_WEIGHTS["cnn_bilstm"]
        self.w_vit = cfg.ENSEMBLE_WEIGHTS["vit"]

    def forward(
        self,
        frames: torch.Tensor,    # (B, T, C, H, W) для CNN
        image:  torch.Tensor,    # (B, C, H, W)    для ViT (средний кадр)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict с логитами и вероятностями каждой модели + ансамбля
        """
        cnn_logits, attn = self.cnn_model(frames)
        vit_logits       = self.vit_model(image)

        cnn_probs = F.softmax(cnn_logits, dim=-1)
        vit_probs = F.softmax(vit_logits, dim=-1)

        ensemble_probs = self.w_cnn * cnn_probs + self.w_vit * vit_probs

        return {
            "cnn_logits":      cnn_logits,
            "vit_logits":      vit_logits,
            "cnn_probs":       cnn_probs,
            "vit_probs":       vit_probs,
            "ensemble_probs":  ensemble_probs,
            "attn_weights":    attn,           # (B, T) — временные веса
        }


# ─────────────────────────────────────────────────────────────────────────────
# Фабричные функции
# ─────────────────────────────────────────────────────────────────────────────
def build_cnn_bilstm() -> DeepfakeCNNBiLSTM:
    model = DeepfakeCNNBiLSTM().to(cfg.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"CNN-BiLSTM модель: {n_params:,} обучаемых параметров")
    return model


def build_vit() -> DeepfakeViT:
    model = DeepfakeViT().to(cfg.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"ViT модель: {n_params:,} обучаемых параметров")
    return model


def build_ensemble(cnn_ckpt: str = None, vit_ckpt: str = None) -> DeepfakeEnsemble:
    cnn = build_cnn_bilstm()
    vit = build_vit()

    if cnn_ckpt and os.path.exists(cnn_ckpt):
        cnn.load_state_dict(torch.load(cnn_ckpt, map_location=cfg.DEVICE))
        log.info(f"CNN-BiLSTM веса загружены: {cnn_ckpt}")

    if vit_ckpt and os.path.exists(vit_ckpt):
        vit.load_state_dict(torch.load(vit_ckpt, map_location=cfg.DEVICE))
        log.info(f"ViT веса загружены: {vit_ckpt}")

    ensemble = DeepfakeEnsemble(cnn, vit).to(cfg.DEVICE)
    return ensemble


if __name__ == "__main__":
    # Проверка архитектур на случайных данных
    print("=" * 55)
    print("ПРОВЕРКА АРХИТЕКТУР")
    print("=" * 55)

    batch_size = 2
    T = cfg.NUM_FRAMES

    # CNN-BiLSTM
    cnn_model = build_cnn_bilstm()
    cnn_model.eval()
    dummy_video = torch.randn(batch_size, T, 3, *cfg.IMAGE_SIZE).to(cfg.DEVICE)
    with torch.no_grad():
        logits, weights = cnn_model(dummy_video)
    print(f"\nCNN-BiLSTM:")
    print(f"  Вход:        {dummy_video.shape}")
    print(f"  Логиты:      {logits.shape}  → {logits}")
    print(f"  Веса вним.:  {weights.shape} (сумма: {weights.sum(-1).tolist()})")

    # ViT
    vit_model = build_vit()
    vit_model.eval()
    dummy_img = torch.randn(batch_size, 3, *cfg.IMAGE_SIZE).to(cfg.DEVICE)
    with torch.no_grad():
        vit_logits = vit_model(dummy_img)
    print(f"\nViT:")
    print(f"  Вход:    {dummy_img.shape}")
    print(f"  Логиты:  {vit_logits.shape} → {vit_logits}")

    print("\n✅ Обе модели работают корректно")
