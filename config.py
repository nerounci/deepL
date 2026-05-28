"""
config.py — Централизованная конфигурация проекта Deepfake Detection
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Пути
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW     = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC    = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINTS  = os.path.join(BASE_DIR, "checkpoints")
OUTPUTS      = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR    = os.path.join(OUTPUTS, "plots")

# ─────────────────────────────────────────────────────────────────────────────
# Данные
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE_DATASET     = "xhlulu/140k-real-and-fake-faces"
CELEB_DF_URL       = "https://github.com/yuezunli/celeb-deepfakeforensics"
FF_DATASET_FORM    = "https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EqLrmco2Cs_H7zg/viewform"

# Разбивка датасета
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# Предобработка изображений
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_SIZE    = (224, 224)   # вход для обеих моделей
FACE_MARGIN   = 0.20         # отступ вокруг лица (20%)
MEAN          = [0.485, 0.456, 0.406]   # ImageNet mean (RGB)
STD           = [0.229, 0.224, 0.225]   # ImageNet std

# Для видео-модели (CNN-BiLSTM)
NUM_FRAMES    = 16           # количество кадров из одного видео
FRAME_STRIDE  = 5            # шаг между кадрами

# ─────────────────────────────────────────────────────────────────────────────
# Аугментация (только для train)
# ─────────────────────────────────────────────────────────────────────────────
AUG_PROB            = 0.5
HORIZONTAL_FLIP_P   = 0.5
BRIGHTNESS_LIMIT    = 0.2
CONTRAST_LIMIT      = 0.2
ROTATION_LIMIT      = 15     # градусов
JPEG_QUALITY_MIN    = 70     # симуляция сжатия

# ─────────────────────────────────────────────────────────────────────────────
# Модель CNN + BiLSTM
# ─────────────────────────────────────────────────────────────────────────────
CNN_BACKBONE        = "efficientnet_b0"
FEATURE_DIM         = 1280         # выход EfficientNet-B0
LSTM_HIDDEN         = 256
LSTM_LAYERS         = 2
LSTM_DROPOUT        = 0.3
ATTENTION_DIM       = 128
NUM_CLASSES         = 2            # real / fake

# ─────────────────────────────────────────────────────────────────────────────
# Модель Vision Transformer
# ─────────────────────────────────────────────────────────────────────────────
VIT_MODEL_NAME      = "vit_base_patch16_224"
VIT_PRETRAINED      = True
VIT_DROP_RATE       = 0.1
VIT_ATTN_DROP_RATE  = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# Обучение
# ─────────────────────────────────────────────────────────────────────────────
SEED            = 42
BATCH_SIZE_CNN  = 8     # меньше из-за sequence × batch памяти
BATCH_SIZE_VIT  = 16
NUM_EPOCHS      = 20
LR_CNN          = 3e-4
LR_VIT          = 1e-4
WEIGHT_DECAY    = 1e-4
SCHEDULER       = "cosine"    # cosine / step
WARMUP_EPOCHS   = 2
EARLY_STOP      = 5           # терпение для early stopping
GRAD_CLIP       = 1.0
MIXED_PREC      = True        # использовать AMP (float16)

# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {"cnn_bilstm": 0.45, "vit": 0.55}   # ViT чуть точнее
DECISION_THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Интерпретация / Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────
GRADCAM_TARGET_LAYER_CNN = "cnn.features[-1]"   # последний conv слой
GRADCAM_TARGET_LAYER_VIT = "blocks[-1].norm1"   # последний Transformer блок

# LLM объяснение (через Anthropic API)
LLM_MODEL   = "claude-sonnet-4-20250514"
LLM_MAX_TOK = 512

# ─────────────────────────────────────────────────────────────────────────────
# Устройство
# ─────────────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4 if DEVICE == "cuda" else 0
PIN_MEMORY  = DEVICE == "cuda"
