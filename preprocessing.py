"""
preprocessing.py — Шаг 2: Предобработка данных
═══════════════════════════════════════════════════════════════════════════════
Пайплайн предобработки:
  1. Face Detection (MTCNN)       — находим лицо, обрезаем с отступом
  2. Alignment                    — выравниваем по ключевым точкам (landmarks)
  3. Quality Filter               — отфильтровываем размытые/маленькие лица
  4. Augmentation (train only)    — аугментация для регуляризации
  5. Normalization                — ImageNet mean/std
  6. Frame Extraction (видео)     — извлечение N равномерных кадров
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as cfg

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Face Detector (MTCNN wrapper)
# ─────────────────────────────────────────────────────────────────────────────
class FaceDetector:
    """
    Обёртка над MTCNN (Multi-task Cascaded CNN) для детекции лиц.
    
    Архитектура MTCNN:
      P-Net (12×12) → R-Net (24×24) → O-Net (48×48)
      Каждая сеть уточняет bounding box и возвращает 5 landmarks.
    """

    def __init__(self, device: str = cfg.DEVICE, min_face_size: int = 40):
        self.device = device
        self.min_face_size = min_face_size
        self._mtcnn = None

    def _load(self):
        if self._mtcnn is None:
            try:
                from facenet_pytorch import MTCNN
                self._mtcnn = MTCNN(
                    image_size=cfg.IMAGE_SIZE[0],
                    margin=int(cfg.IMAGE_SIZE[0] * cfg.FACE_MARGIN),
                    min_face_size=self.min_face_size,
                    thresholds=[0.6, 0.7, 0.7],   # P-Net, R-Net, O-Net пороги
                    factor=0.709,                   # масштабный шаг пирамиды
                    post_process=False,             # возвращаем uint8
                    device=self.device,
                    keep_all=False,                 # только самое большое лицо
                    select_largest=True,
                )
                log.info("MTCNN загружен")
            except ImportError:
                log.warning("facenet-pytorch не установлен. Используем OpenCV Haar.")
                self._mtcnn = "haar"

    def extract_face(
        self,
        image: np.ndarray,
        return_landmarks: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Извлекает лицо из изображения.

        Args:
            image: RGB изображение (H, W, 3) uint8
            return_landmarks: вернуть ключевые точки?

        Returns:
            Обрезанное лицо (224, 224, 3) или None
        """
        self._load()
        pil_img = Image.fromarray(image)

        if self._mtcnn == "haar":
            return self._haar_fallback(image)

        try:
            face_tensor, prob = self._mtcnn(pil_img, return_prob=True)
            if face_tensor is None or prob < 0.90:
                return None
            # Конвертируем tensor (C, H, W) → numpy (H, W, C)
            face = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            return face
        except Exception as e:
            log.debug(f"MTCNN ошибка: {e}. Fallback к Haar.")
            return self._haar_fallback(image)

    def _haar_fallback(self, image: np.ndarray) -> Optional[np.ndarray]:
        """OpenCV Haar Cascade как запасной вариант."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        if len(faces) == 0:
            return None

        # Берём самое большое лицо
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        margin = int(max(w, h) * cfg.FACE_MARGIN)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, cfg.IMAGE_SIZE)
        return face


# ─────────────────────────────────────────────────────────────────────────────
# 2. Quality Filter
# ─────────────────────────────────────────────────────────────────────────────
class QualityFilter:
    """
    Фильтрует изображения низкого качества.

    Методы:
      - Laplacian variance: обнаруживает смаз (blur detection)
      - Brightness check: отфильтровываем чёрные / пересвеченные
    """

    def __init__(
        self,
        blur_threshold: float = 50.0,
        brightness_min: float = 20.0,
        brightness_max: float = 235.0,
    ):
        self.blur_thr = blur_threshold
        self.br_min = brightness_min
        self.br_max = brightness_max

    def is_good(self, image: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """
        Проверяет качество изображения.

        Returns:
            (is_good: bool, metrics: dict)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Лапласов вариационный критерий (чем выше — тем резче)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Средняя яркость
        brightness = gray.mean()

        metrics = {
            "laplacian_var": float(laplacian_var),
            "brightness":    float(brightness),
        }

        ok = (
            laplacian_var >= self.blur_thr
            and self.br_min <= brightness <= self.br_max
        )
        return ok, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3. Augmentation Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def build_transforms(split: str) -> T.Compose:
    """
    Создаёт трансформации для split в {train, val, test}.

    Train: агрессивная аугментация для регуляризации
    Val/Test: только ресайз + нормализация

    Важно для дипфейков: не используем crop, который может удалить артефакты
    на краях лица — ключевые признаки для детекции.
    """
    mean = cfg.MEAN
    std  = cfg.STD

    if split == "train":
        return T.Compose([
            T.Resize(cfg.IMAGE_SIZE),
            T.RandomHorizontalFlip(p=cfg.HORIZONTAL_FLIP_P),
            T.ColorJitter(
                brightness=cfg.BRIGHTNESS_LIMIT,
                contrast=cfg.CONTRAST_LIMIT,
                saturation=0.1,
                hue=0.05,
            ),
            T.RandomRotation(degrees=cfg.ROTATION_LIMIT),
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),  # небольшой сдвиг
                scale=(0.95, 1.05),
            ),
            # JPEG компрессионные артефакты — важны для реализма
            T.RandomApply([
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            # Случайное стирание (Cutout) — улучшает робастность
            T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return T.Compose([
            T.Resize(cfg.IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Video Frame Extractor
# ─────────────────────────────────────────────────────────────────────────────
class VideoFrameExtractor:
    """
    Извлекает N равномерно распределённых кадров из видеофайла.
    Используется для CNN-BiLSTM (темпоральная модель).
    """

    def __init__(self, num_frames: int = cfg.NUM_FRAMES):
        self.num_frames = num_frames
        self.face_detector = FaceDetector()

    def extract(self, video_path: str) -> Optional[List[np.ndarray]]:
        """
        Извлекает кадры из видео с детекцией лица.

        Returns:
            Список из num_frames массивов (224, 224, 3) или None
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.warning(f"Не удалось открыть видео: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            log.warning(f"Мало кадров ({total_frames}) в {video_path}")
            cap.release()
            return None

        # Равномерные индексы кадров
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = self.face_detector.extract_face(frame_rgb)

            if face is not None:
                frames.append(face)

        cap.release()

        if len(frames) < self.num_frames // 2:
            log.warning(f"Слишком мало лиц найдено в {video_path}")
            return None

        # Дополняем повтором последнего кадра, если не хватает
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames[:self.num_frames]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dataset классы
# ─────────────────────────────────────────────────────────────────────────────
class DeepfakeImageDataset(Dataset):
    """
    PyTorch Dataset для изображений (ViT модель).
    Структура: dataset_dir/{split}/{real,fake}/*.jpg
    """

    def __init__(self, root_dir: str, split: str = "train"):
        self.transform = build_transforms(split)
        self.samples: List[Tuple[str, int]] = []

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = Path(root_dir) / split / cls
            if cls_dir.exists():
                for img_path in sorted(cls_dir.glob("*.jpg")):
                    self.samples.append((str(img_path), label))
            else:
                log.warning(f"Директория не найдена: {cls_dir}")

        log.info(f"Dataset[{split}]: {len(self.samples)} samples "
                 f"(real={sum(1 for _,l in self.samples if l==0)}, "
                 f"fake={sum(1 for _,l in self.samples if l==1)})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label


class DeepfakeVideoDataset(Dataset):
    """
    PyTorch Dataset для видео последовательностей (CNN-BiLSTM модель).
    Возвращает тензор (num_frames, C, H, W).
    """

    def __init__(self, frame_dir: str, split: str = "train"):
        self.transform = build_transforms(split)
        self.sequences: List[Tuple[List[str], int]] = []

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = Path(frame_dir) / split / cls
            if not cls_dir.exists():
                continue

            # Группируем кадры по видео (по префиксу имени файла)
            video_frames: Dict[str, List[Path]] = {}
            for frame in sorted(cls_dir.glob("*.jpg")):
                # Формат имени: videoID_frame_N.jpg
                vid_id = "_".join(frame.stem.split("_")[:-1])
                video_frames.setdefault(vid_id, []).append(frame)

            for vid_id, frames in video_frames.items():
                frames_sorted = sorted(frames)
                if len(frames_sorted) >= cfg.NUM_FRAMES:
                    paths = [str(f) for f in frames_sorted[:cfg.NUM_FRAMES]]
                    self.sequences.append((paths, label))

        log.info(f"VideoDataset[{split}]: {len(self.sequences)} видео-последовательностей")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        frame_paths, label = self.sequences[idx]
        frames = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.transform(img))
        # (num_frames, C, H, W)
        return torch.stack(frames, dim=0), label


# ─────────────────────────────────────────────────────────────────────────────
# 6. Полный пайплайн предобработки
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_dataset(raw_dir: str, out_dir: str):
    """
    Обрабатывает сырой датасет:
      1. Детекция лица → обрезка
      2. Фильтрация качества
      3. Сохранение в out_dir

    Выводит статистику фильтрации.
    """
    detector = FaceDetector()
    quality  = QualityFilter()
    raw      = Path(raw_dir)
    out      = Path(out_dir)

    stats = {"processed": 0, "no_face": 0, "low_quality": 0, "saved": 0}

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            src = raw / split / cls
            dst = out / split / cls
            dst.mkdir(parents=True, exist_ok=True)

            if not src.exists():
                continue

            img_paths = list(src.glob("*.jpg"))
            for path in tqdm(img_paths, desc=f"  {split}/{cls}", leave=False):
                stats["processed"] += 1

                img_arr = np.array(Image.open(path).convert("RGB"))

                # Детекция лица
                face = detector.extract_face(img_arr)
                if face is None:
                    # Если лицо не найдено, используем исходное изображение (ресайз)
                    face = cv2.resize(img_arr, cfg.IMAGE_SIZE)
                    stats["no_face"] += 1

                # Фильтрация качества
                ok, metrics = quality.is_good(face)
                if not ok:
                    stats["low_quality"] += 1
                    # Не отбрасываем, просто логируем
                    pass

                # Сохраняем
                Image.fromarray(face).save(dst / path.name, "JPEG", quality=95)
                stats["saved"] += 1

    # Итог
    log.info("\n" + "=" * 50)
    log.info("РЕЗУЛЬТАТЫ ПРЕДОБРАБОТКИ")
    log.info("=" * 50)
    log.info(f"  Обработано:       {stats['processed']}")
    log.info(f"  Лицо не найдено:  {stats['no_face']} ({stats['no_face']/max(1,stats['processed'])*100:.1f}%)")
    log.info(f"  Низкое качество:  {stats['low_quality']} ({stats['low_quality']/max(1,stats['processed'])*100:.1f}%)")
    log.info(f"  Сохранено:        {stats['saved']}")
    log.info("=" * 50)

    return stats


def visualize_preprocessing(raw_dir: str, save_path: str = None):
    """
    Визуализирует этапы предобработки одного изображения.
    Показывает: исходное → детекция лица → аугментация → нормализация.
    """
    import matplotlib.pyplot as plt

    raw = Path(raw_dir)

    # Берём по одному примеру из каждого класса
    examples = []
    for cls, label in [("real", 0), ("fake", 1)]:
        for split in ["train", "val", "test"]:
            imgs = list((raw / split / cls).glob("*.jpg"))
            if imgs:
                examples.append((str(imgs[0]), cls, label))
                break

    fig, axes = plt.subplots(len(examples), 4, figsize=(16, 4 * len(examples)))
    fig.suptitle("Пайплайн предобработки", fontsize=14, fontweight="bold")

    stages = ["1. Исходное", "2. Детекция лица", "3. Аугментация", "4. Нормализованное"]
    detector = FaceDetector()
    aug = build_transforms("train")
    norm = build_transforms("val")

    for row, (path, cls, label) in enumerate(examples):
        img_orig = np.array(Image.open(path).convert("RGB"))

        # Исходное
        axes[row, 0].imshow(img_orig)
        axes[row, 0].set_title(stages[0])
        axes[row, 0].axis("off")

        # После детекции лица
        face = detector.extract_face(img_orig)
        if face is None:
            face = cv2.resize(img_orig, cfg.IMAGE_SIZE)
        axes[row, 1].imshow(face)
        axes[row, 1].set_title(stages[1])
        axes[row, 1].axis("off")

        # После аугментации (денормализуем для отображения)
        face_pil = Image.fromarray(face)
        aug_tensor = aug(face_pil)
        # Денормализация для визуализации
        aug_vis = aug_tensor.clone()
        for c, (m, s) in enumerate(zip(cfg.MEAN, cfg.STD)):
            aug_vis[c] = aug_vis[c] * s + m
        aug_vis = aug_vis.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[row, 2].imshow(aug_vis)
        axes[row, 2].set_title(stages[2])
        axes[row, 2].axis("off")

        # Нормализованное (heatmap значений)
        norm_tensor = norm(face_pil)
        norm_vis = norm_tensor.mean(dim=0).numpy()
        im = axes[row, 3].imshow(norm_vis, cmap="RdBu_r")
        axes[row, 3].set_title(f"{stages[3]}\n({'REAL' if label==0 else 'FAKE'})")
        axes[row, 3].axis("off")
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save = save_path or os.path.join(cfg.PLOTS_DIR, "02_preprocessing.png")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    log.info(f"Граф предобработки сохранён: {save}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection — Предобработка")
    parser.add_argument("--input_dir",  default=os.path.join(cfg.DATA_RAW, "synthetic_demo"))
    parser.add_argument("--output_dir", default=cfg.DATA_PROC)
    parser.add_argument("--visualize",  action="store_true")
    args = parser.parse_args()

    log.info("Запуск предобработки...")
    preprocess_dataset(args.input_dir, args.output_dir)

    if args.visualize:
        visualize_preprocessing(args.input_dir)


if __name__ == "__main__":
    main()
