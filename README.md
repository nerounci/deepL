# Deepfake Detection System — Deep Learning Project

## Комплексная система обнаружения дипфейков с использованием методов глубокого обучения

---

##  Описание проекта

Система обнаружения дипфейков на основе двух нейросетевых архитектур:

| Модель | Подход | Описание |
|--------|--------|----------|
| **EfficientNet-B0 + BiLSTM + Attention** | RNN + Attention | Анализ временных последовательностей кадров |
| **Vision Transformer (ViT-B/16)** | Transformer | Пространственный анализ артефактов |
| **Ensemble + LLM Explainer** | Генеративные модели | Интерпретация результатов на естественном языке |

**Применяемые подходы (из задания):**
-  **a Рекуррентные нейронные сети** — BiLSTM для темпорального анализа
-  **b Трансформеры и механизмы внимания** — ViT-B/16 + multi-head self-attention
-  **c Генеративные текстовые модели** — LLM-based объяснение результатов детекции

---

## Датасеты

### Основной датасет: FaceForensics++
- **Ссылка:** https://github.com/ondyari/FaceForensics
- **Размер:** ~1.5TB (полный) / ~50GB (сжатый c23)
- **Описание:** 1000 оригинальных + 4000 манипулированных видео (4 метода манипуляции)
- **Доступ:** Заполнить форму на https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EqLrmco2Cs_H7zg/viewform

### Альтернативный Celeb-DF v2
- **Ссылка:** https://github.com/yuezunli/celeb-deepfakeforensics
- **Размер:** ~7GB
- **Описание:** 590 реальных + 5639 дипфейк видео

### Для быстрого прототипа (изображения): 140k Real and Fake Faces
- **Ссылка:** https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- **Размер:** ~300MB
- **Команда:** `kaggle datasets download -d xhlulu/140k-real-and-fake-faces`

### DFDC (DeepFake Detection Challenge)
- **Ссылка:** https://www.kaggle.com/c/deepfake-detection-challenge/data
- **Размер:** ~470GB
- **Описание:** Крупнейший публичный датасет от Facebook

---

##  Установка

```bash
# Клонировать репозиторий
git clone <repo_url>
cd deepfake_detection

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Установить зависимости
pip install -r requirements.txt
```

---

##  Запуск

```bash
# 1. Сбор и подготовка данных
python src/data_collection.py --dataset kaggle_140k --output_dir data/raw

# 2. Предобработка
python src/preprocessing.py --input_dir data/raw --output_dir data/processed

# 3. Обучение модели CNN+BiLSTM
python src/train.py --model cnn_bilstm --epochs 20 --batch_size 8

# 4. Обучение ViT модели
python src/train.py --model vit --epochs 10 --batch_size 16

# 5. Оценка
python src/evaluate.py --checkpoint checkpoints/best_model.pth

# 6. Интерпретация
python src/interpret.py --image_path data/test_face.jpg

# Полный пайплайн через main.py
python main.py --mode full_pipeline
```

---

##  Структура проекта

```
deepfake_detection/
├── README.md
├── requirements.txt
├── main.py                    # Точка входа
├── src/
│   ├── config.py              # Конфигурация
│   ├── data_collection.py     # Сбор данных
│   ├── dataset.py             # PyTorch Dataset классы
│   ├── preprocessing.py       # Предобработка (face detection, augmentation)
│   ├── models.py              # Архитектуры нейросетей
│   ├── train.py               # Обучение
│   ├── evaluate.py            # Метрики и оценка
│   └── interpret.py           # Grad-CAM, attention viz, LLM explain
├── data/
│   ├── raw/                   # Сырые данные
│   └── processed/             # Обработанные данные
├── checkpoints/               # Веса моделей
└── outputs/
    └── plots/                 # Графики и визуализации
```

---

##  Архитектурные схемы

### Модель 1: EfficientNet-B0 + BiLSTM + Attention
```
Video → [Frame1, Frame2, ..., FrameN]
         ↓ EfficientNet-B0 (shared weights)
         [f1, f2, ..., fN]  (1280-dim each)
         ↓ BiLSTM (256 hidden × 2 directions)
         [h1, h2, ..., hN]  (512-dim each)
         ↓ Temporal Attention
         context_vector (512-dim)
         ↓ FC(512→128→2)
         [P(real), P(fake)]
```

### Модель 2: Vision Transformer (ViT-B/16)
```
Image (224×224×3)
↓ Patch Embedding (16×16 patches → 196 tokens + CLS)
↓ 12× Transformer Encoder Blocks
   [Multi-Head Self-Attention + FFN + LayerNorm]
↓ CLS token → FC → [P(real), P(fake)]
```
