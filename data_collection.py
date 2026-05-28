"""
data_collection.py — Шаг 1: Сбор данных
═══════════════════════════════════════════════════════════════════════════════
Поддерживает три источника:
  1. kaggle_140k  — 140k Real and Fake Faces (изображения, ~300MB)
  2. celeb_df     — Celeb-DF v2 (видео, ~7GB)
  3. demo_synthetic — генерация синтетического мини-датасета для демонстрации
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import json
import random
import logging
import argparse
import zipfile
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Описание датасетов (с ссылками)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_REGISTRY = {
    "kaggle_140k": {
        "name":    "140k Real and Fake Faces",
        "source":  "Kaggle",
        "url":     "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces",
        "command": "kaggle datasets download -d xhlulu/140k-real-and-fake-faces",
        "size":    "300 MB",
        "classes": {"real": 70000, "fake": 70000},
        "format":  "JPEG images 256×256",
        "notes":   "Fake изображения сгенерированы StyleGAN2",
    },
    "celeb_df": {
        "name":    "Celeb-DF v2",
        "source":  "GitHub",
        "url":     "https://github.com/yuezunli/celeb-deepfakeforensics",
        "command": "Требуется заполнить форму: https://forms.gle/HKT8MxsoTCbS3mgr9",
        "size":    "7 GB",
        "classes": {"real": 590, "fake": 5639},
        "format":  "MP4 видео",
        "notes":   "High-quality deepfakes знаменитостей",
    },
    "ff++": {
        "name":    "FaceForensics++",
        "source":  "GitHub",
        "url":     "https://github.com/ondyari/FaceForensics",
        "command": "python FaceForensics/dataset/download_FaceForensics.py . -d all -c c23 -t videos",
        "size":    "50 GB (c23 compression)",
        "classes": {"real": 1000, "fake": 4000},
        "format":  "MP4 видео (4 вида манипуляций)",
        "notes":   "DF, F2F, FS, NT манипуляции; требуется регистрация",
    },
    "dfdc": {
        "name":    "DeepFake Detection Challenge",
        "source":  "Kaggle",
        "url":     "https://www.kaggle.com/c/deepfake-detection-challenge/data",
        "command": "kaggle competitions download -c deepfake-detection-challenge",
        "size":    "470 GB",
        "classes": {"real": "~23k", "fake": "~100k"},
        "format":  "MP4 видео",
        "notes":   "Крупнейший публичный датасет, Facebook AI",
    },
}


def print_dataset_info():
    """Выводит информацию о доступных датасетах."""
    log.info("=" * 65)
    log.info("ДОСТУПНЫЕ ДАТАСЕТЫ ДЛЯ ОБНАРУЖЕНИЯ ДИПФЕЙКОВ")
    log.info("=" * 65)
    for key, info in DATASET_REGISTRY.items():
        log.info(f"\n[{key.upper()}] {info['name']}")
        log.info(f"  Источник : {info['source']}")
        log.info(f"  URL      : {info['url']}")
        log.info(f"  Размер   : {info['size']}")
        log.info(f"  Классы   : {info['classes']}")
        log.info(f"  Формат   : {info['format']}")
        log.info(f"  Команда  : {info['command']}")
        log.info(f"  Заметки  : {info['notes']}")
    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Загрузка Kaggle датасета
# ─────────────────────────────────────────────────────────────────────────────
def download_kaggle_dataset(output_dir: str) -> str:
    """
    Загружает 140k Real and Fake Faces с Kaggle.
    Требует: ~/.kaggle/kaggle.json с API-ключом.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Загрузка датасета с Kaggle: xhlulu/140k-real-and-fake-faces")
    log.info("Убедитесь, что ~/.kaggle/kaggle.json настроен корректно.")
    log.info("Инструкция: https://www.kaggle.com/docs/api")

    zip_path = out / "140k-real-and-fake-faces.zip"
    if zip_path.exists():
        log.info("Архив уже скачан, пропускаем загрузку.")
    else:
        cmd = f"kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p {out}"
        log.info(f"Выполняем: {cmd}")
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"Ошибка Kaggle CLI: {result.stderr}")
            log.info("Переключаемся на синтетический демо-датасет...")
            return generate_synthetic_demo(output_dir)

    # Распаковка
    log.info("Распаковка архива...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)

    extracted = out / "real_vs_fake" / "real-vs-fake"
    log.info(f"Датасет распакован в {extracted}")
    return str(extracted)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Генерация синтетического демо-датасета
# ─────────────────────────────────────────────────────────────────────────────
def _generate_face(size: int = 256, is_fake: bool = False) -> Image.Image:
    """
    Создаёт синтетическое изображение лица.
    Real:  гладкие тона кожи
    Fake:  добавляем характерные артефакты дипфейков:
           - неестественные переходы цвета
           - размытые края (проблема blending mask)
           - артефакты сетки (grid artifacts из conv-транспозиций)
    """
    rng = np.random.RandomState()

    # Базовый тон кожи (реалистичный диапазон)
    skin_r = rng.randint(180, 230)
    skin_g = rng.randint(140, 180)
    skin_b = rng.randint(110, 150)

    img_arr = np.zeros((size, size, 3), dtype=np.uint8)

    # Маска лица (эллипс)
    cx, cy = size // 2, size // 2
    for y in range(size):
        for x in range(size):
            if ((x - cx) / (size * 0.35))**2 + ((y - cy) / (size * 0.45))**2 < 1:
                noise = rng.randint(-15, 15, 3)
                img_arr[y, x] = np.clip(
                    [skin_r + noise[0], skin_g + noise[1], skin_b + noise[2]],
                    0, 255
                )

    img = Image.fromarray(img_arr)

    if is_fake:
        # Артефакт 1: неоднородный блендинг на краях лица
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([cx - 80, cy - 100, cx + 80, cy + 100], fill=200)
        blurred_mask = mask.filter(ImageFilter.GaussianBlur(8))

        # Артефакт 2: grid artifact (следы от транспонированных свёрток)
        grid_arr = np.array(img)
        step = 16
        for i in range(0, size, step):
            grid_arr[i, :] = np.clip(grid_arr[i, :].astype(int) + rng.randint(-20, 20), 0, 255)
            grid_arr[:, i] = np.clip(grid_arr[:, i].astype(int) + rng.randint(-20, 20), 0, 255)
        img = Image.fromarray(grid_arr.astype(np.uint8))

        # Артефакт 3: неестественный цветовой сдвиг (несоответствие цветовых пространств)
        r, g, b = img.split()
        r = r.point(lambda p: min(255, p + rng.randint(5, 25)))
        img = Image.merge("RGB", (r, g, b))

        # Артефакт 4: размытие краёв (проблема граничных пикселей)
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))

    return img


def generate_synthetic_demo(
    output_dir: str,
    n_real: int = 500,
    n_fake: int = 500,
) -> str:
    """
    Генерирует синтетический датасет для демонстрации пайплайна.
    Создаёт n_real реальных и n_fake дипфейк изображений с артефактами.
    """
    log.info(f"Генерация синтетического датасета: {n_real} real / {n_fake} fake")
    base = Path(output_dir) / "synthetic_demo"

    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    classes = {"real": n_real, "fake": n_fake}

    metadata = []

    for split_name, ratio in splits.items():
        for cls, total in classes.items():
            split_dir = base / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)

            n = int(total * ratio)
            for i in tqdm(range(n), desc=f"{split_name}/{cls}", leave=False):
                img = _generate_face(size=256, is_fake=(cls == "fake"))
                path = split_dir / f"{cls}_{split_name}_{i:04d}.jpg"
                img.save(path, "JPEG", quality=95)
                metadata.append({
                    "path":  str(path.relative_to(base)),
                    "label": 0 if cls == "real" else 1,
                    "split": split_name,
                    "class": cls,
                })

    # Сохраняем метаданные
    meta_path = base / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Статистика
    _log_dataset_stats(metadata)

    log.info(f"Датасет создан: {base}")
    return str(base)


def _log_dataset_stats(metadata: list):
    """Выводит статистику по датасету."""
    from collections import Counter
    splits = Counter((m["split"], m["class"]) for m in metadata)

    log.info("\n" + "=" * 45)
    log.info("СТАТИСТИКА ДАТАСЕТА")
    log.info("=" * 45)
    log.info(f"{'Сплит':<10} {'Real':>8} {'Fake':>8} {'Total':>8}")
    log.info("-" * 45)

    totals = {"train": {}, "val": {}, "test": {}}
    for (split, cls), cnt in splits.items():
        totals[split][cls] = cnt

    grand_total = 0
    for split in ["train", "val", "test"]:
        real = totals[split].get("real", 0)
        fake = totals[split].get("fake", 0)
        total = real + fake
        grand_total += total
        log.info(f"{split:<10} {real:>8} {fake:>8} {total:>8}")

    log.info("-" * 45)
    log.info(f"{'ИТОГО':<10} {sum(v.get('real',0) for v in totals.values()):>8} "
             f"{sum(v.get('fake',0) for v in totals.values()):>8} {grand_total:>8}")
    log.info("=" * 45)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Анализ EDA (Exploratory Data Analysis)
# ─────────────────────────────────────────────────────────────────────────────
def exploratory_data_analysis(dataset_dir: str, save_path: str = None):
    """
    Базовый EDA: распределение классов, примеры изображений,
    статистика яркости/шума.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    base = Path(dataset_dir)
    all_images = {"real": [], "fake": []}

    for cls in ["real", "fake"]:
        for split in ["train", "val", "test"]:
            d = base / split / cls
            if d.exists():
                all_images[cls].extend(list(d.glob("*.jpg"))[:200])

    log.info(f"EDA: real={len(all_images['real'])}, fake={len(all_images['fake'])}")

    # Подсчёт статистик пикселей
    stats = {}
    for cls, paths in all_images.items():
        values = []
        for p in paths[:100]:
            arr = np.array(Image.open(p).convert("RGB"))
            values.append(arr.mean())
        stats[cls] = np.array(values)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle("EDA: Распределение данных и характеристики изображений", fontsize=14, fontweight="bold")

    # 1. Баланс классов (train)
    ax1 = fig.add_subplot(gs[0, 0])
    splits_count = {}
    for split in ["train", "val", "test"]:
        r = len(list((base / split / "real").glob("*.jpg"))) if (base / split / "real").exists() else 0
        f = len(list((base / split / "fake").glob("*.jpg"))) if (base / split / "fake").exists() else 0
        splits_count[split] = {"Real": r, "Fake": f}

    x = np.arange(3)
    w = 0.35
    splits = list(splits_count.keys())
    reals = [splits_count[s]["Real"] for s in splits]
    fakes = [splits_count[s]["Fake"] for s in splits]
    ax1.bar(x - w/2, reals, w, label="Real", color="#2196F3", alpha=0.85)
    ax1.bar(x + w/2, fakes, w, label="Fake", color="#F44336", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.set_title("Распределение по сплитам")
    ax1.set_ylabel("Количество изображений")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 2. Гистограмма яркости
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(stats["real"],  bins=25, alpha=0.7, color="#2196F3", label="Real")
    ax2.hist(stats["fake"],  bins=25, alpha=0.7, color="#F44336", label="Fake")
    ax2.set_title("Распределение средней яркости")
    ax2.set_xlabel("Средняя яркость пикселя")
    ax2.set_ylabel("Частота")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Круговая диаграмма
    ax3 = fig.add_subplot(gs[0, 2])
    total_r = sum(r for r in reals)
    total_f = sum(f for f in fakes)
    ax3.pie([total_r, total_f],
            labels=["Real", "Fake"],
            colors=["#2196F3", "#F44336"],
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.05, 0.05))
    ax3.set_title(f"Баланс датасета\n(total={total_r+total_f})")

    # 4–6. Примеры изображений (3 real + 3 fake)
    for idx, (cls, color, row) in enumerate([("real", "#2196F3", 1), ("fake", "#F44336", 1)]):
        paths = all_images[cls][:3]
        for j, p in enumerate(paths):
            ax = fig.add_subplot(gs[1, j])
            img = Image.open(p).resize((128, 128))
            ax.imshow(img)
            label = "REAL ✓" if cls == "real" else "FAKE ✗"
            ax.set_title(f"{label}", color=color, fontweight="bold")
            ax.axis("off")
            if idx == 1:
                # Добавляем overlay для fake
                ax.set_title(f"FAKE ✗ (пример {j+1})", color="#F44336", fontweight="bold")

    plt.tight_layout()
    save = save_path or os.path.join(cfg.PLOTS_DIR, "01_eda.png")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    log.info(f"EDA график сохранён: {save}")
    plt.close()
    return save


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection — Сбор данных")
    parser.add_argument("--dataset",    default="demo_synthetic",
                        choices=["kaggle_140k", "demo_synthetic"],
                        help="Источник данных")
    parser.add_argument("--output_dir", default=cfg.DATA_RAW)
    parser.add_argument("--n_real",     type=int, default=500,
                        help="Кол-во real изображений (для demo)")
    parser.add_argument("--n_fake",     type=int, default=500,
                        help="Кол-во fake изображений (для demo)")
    parser.add_argument("--eda",        action="store_true",
                        help="Запустить EDA после сбора данных")
    args = parser.parse_args()

    print_dataset_info()

    if args.dataset == "kaggle_140k":
        dataset_dir = download_kaggle_dataset(args.output_dir)
    else:
        dataset_dir = generate_synthetic_demo(args.output_dir, args.n_real, args.n_fake)

    if args.eda:
        exploratory_data_analysis(dataset_dir)

    log.info(f"\nДатасет готов: {dataset_dir}")
    return dataset_dir


if __name__ == "__main__":
    main()
