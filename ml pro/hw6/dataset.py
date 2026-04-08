import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
from utils import rle_decode
from config import CLASSES, IMG_SIZE


def get_transforms(phase: str) -> A.Compose:
    """Возвращает pipeline аугментаций для указанной фазы обучения.

    Train: агрессивные аугментации увеличивают разнообразие обучающих примеров
           и снижают переобучение.
    Val/Test: только детерминированные преобразования — ресайз и нормализация.

    Args:
        phase: 'train' (с аугментациями) или 'val'/'test' (без аугментаций).

    Returns:
        albumentations.Compose pipeline.
    """
    if phase == "train":
        return A.Compose([
            # Ресайз всегда идёт первым: гарантирует H=320, W=480 до всех аугментаций
            A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),    # 1400x2100 → 320x480
            A.HorizontalFlip(p=0.5),                            # горизонтальное зеркалирование (50% вероятность)
            A.VerticalFlip(p=0.5),                              # вертикальное зеркалирование
            # RandomRotate90 поворот на 90/270° меняет H и W местами (320→480, 480→320),
            # что приводит к RuntimeError в DataLoader (stack expects equal size).
            # Замена: Affine с rotate=±90° сохраняет размер (внутри добавляет поля)
            A.Affine(                                           # аффинные преобразования
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # ±5% сдвига
                scale=(0.9, 1.1),                               # ±10% масштабирования
                rotate=(-90, 90),                               # ±90° поворот (не меняет H/W)
                p=0.5
            ),
            A.RandomBrightnessContrast(     # случайное изменение яркости и контраста
                brightness_limit=0.2,       # ±20% яркости
                contrast_limit=0.2,         # ±20% контраста
                p=0.4
            ),
            A.GaussNoise(p=0.3),            # гауссов шум (имитация сенсорного шума камеры)
            A.CoarseDropout(                # Cutout: случайные прямоугольные "дыры"
                hole_height_range=(0.02, 0.15),   # 2-15% высоты изображения
                hole_width_range=(0.02, 0.15),    # 2-15% ширины
                num_holes_range=(1, 8),           # 1-8 дыр
                p=0.3
            ),
            A.Normalize(                    # нормализация на статистику ImageNet
                mean=(0.485, 0.456, 0.406), # средние значения каналов R, G, B
                std=(0.229, 0.224, 0.225)   # стандартные отклонения каналов R, G, B
            ),
            ToTensorV2(),                   # конвертируем HWC numpy uint8 → CHW torch.FloatTensor
        ])
    else:  # val / test — только детерминированные преобразования
        return A.Compose([
            A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class CloudDataset(Dataset):
    """PyTorch Dataset для многоклассовой сегментации облачных образований.

    Читает спутниковые снимки и декодирует RLE-маски из CSV-файла.
    Поддерживает 4 класса: Fish, Flower, Gravel, Sugar.

    Структура CSV (train.csv):
        Image_Label: '<image_filename>_<class>' (например, '0011165.jpg_Fish')
        EncodedPixels: RLE-строка маски или NaN если объекта нет на снимке

    Args:
        df: DataFrame с колонками Image_Label и EncodedPixels.
        img_dir: путь к директории с изображениями.
        phase: 'train' или 'val' — определяет набор аугментаций.

    Returns (per __getitem__):
        image : FloatTensor (3, H, W) — нормализованное изображение.
        masks : FloatTensor (4, H, W) — бинарные маски, по одному каналу на класс.
    """

    ORIG_H, ORIG_W = 1400, 2100  # оригинальный размер спутниковых снимков

    def __init__(self, df: pd.DataFrame, img_dir: str, phase: str = "train"):
        self.img_dir    = img_dir
        self.transforms = get_transforms(phase)

        # Преобразуем «длинный» формат CSV в «широкий»:
        # Исходно: много строк (по одной на пару image+class)
        # После pivot: одна строка на изображение, 4 колонки с RLE масками
        df = df.copy()
        df[["image_id", "class"]] = df["Image_Label"].str.rsplit(
            "_", n=1, expand=True  # разделяем справа на 2 части по последнему «_»
        )
        self.df = (
            df.pivot(index="image_id", columns="class", values="EncodedPixels")
              .reset_index()
        )
        # Гарантируем наличие всех 4 колонок классов
        # (некоторые классы могут отсутствовать в конкретном сплите)
        for c in CLASSES:
            if c not in self.df.columns:
                self.df[c] = np.nan

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])

        # Загружаем изображение как RGB numpy-массив (H, W, 3) dtype=uint8
        image = np.array(Image.open(img_path).convert("RGB"))

        # Декодируем RLE каждого класса и собираем в 3D массив (H, W, 4)
        # Каждый срез masks[..., i] — бинарная маска i-го класса
        masks = np.stack(
            [rle_decode(row[c], (self.ORIG_H, self.ORIG_W)) for c in CLASSES],
            axis=-1,
        ).astype(np.float32)  # конвертируем в float32 для PyTorch

        # Albumentations принимает список 2D-масок и синхронно применяет
        # одинаковые случайные преобразования к изображению и всем маскам
        aug   = self.transforms(image=image, masks=[masks[..., i] for i in range(4)])
        image = aug["image"]  # FloatTensor (3, H, W) после ToTensorV2

        # ToTensorV2 в новых версиях albumentations (1.4+) может вернуть маски
        # как torch.Tensor вместо np.ndarray — обрабатываем оба случая
        processed_masks = []
        for m in aug["masks"]:
            if isinstance(m, torch.Tensor):
                processed_masks.append(m.float())          # уже Tensor — только приводим к float32
            else:
                processed_masks.append(torch.from_numpy(np.array(m)).float())  # numpy → Tensor

        masks = torch.stack(processed_masks, dim=0)  # FloatTensor (4, H, W)

        return image, masks
