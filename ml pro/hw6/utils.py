import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import PLOT_DIR


def set_seed(seed: int = 42) -> None:
    """Фиксирует все генераторы случайных чисел для воспроизводимости результатов.

    Устанавливает одинаковое seed для:
    - Python random
    - NumPy random
    - PyTorch CPU и GPU
    - cuDNN (детерминированный режим)

    Args:
        seed: значение зерна (по умолчанию 42).
    """
    random.seed(seed)          # Python random
    np.random.seed(seed)       # NumPy
    torch.manual_seed(seed)    # CPU PyTorch
    torch.cuda.manual_seed_all(seed)           # все GPU
    torch.backends.cudnn.deterministic = True  # детерминированные алгоритмы cuDNN
    torch.backends.cudnn.benchmark     = False # отключаем автотюнинг (мешает воспроизводимости)


def rle_decode(mask_rle: str, shape: tuple) -> np.ndarray:
    """Декодирует RLE-строку (Run-Length Encoding) в бинарную маску.

    RLE — формат, где маска представлена парами (start, length) пикселей
    в порядке column-major (Fortran order).
    Пример: "1 3 10 5" -> пиксели 1,2,3 и 10,11,12,13,14 равны 1.

    Args:
        mask_rle: RLE-строка или пустое/NaN значение (нет маски).
        shape: (H, W) результирующего массива.

    Returns:
        Бинарный массив uint8 формы (H, W).
    """
    # Если маски нет — возвращаем нулевой массив
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Чётные позиции (0,2,4,...) — начала отрезков, нечётные (1,3,5,...) — длины
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1  # Kaggle использует 1-индексацию -> конвертируем в 0-индексацию
    ends = starts + lengths

    # Строим плоский массив и заполняем 1 в нужных диапазонах
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # Reshape в column-major (order='F') — именно так записан Kaggle RLE
    return img.reshape(shape, order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Кодирует бинарную маску в RLE-строку.

    Args:
        mask: бинарный массив (H, W) из значений 0 и 1.

    Returns:
        RLE-строка в формате Kaggle (1-индексация, column-major).
    """
    pixels = mask.flatten(order="F")  # разворачиваем в column-major порядке
    # Добавляем нули по краям для корректного подсчёта переходов 0->1 и 1->0
    pixels = np.concatenate([[0], pixels, [0]])
    # Находим позиции переходов между 0 и 1
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Нечётные элементы runs — длины отрезков единиц
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


class EarlyStopping:
    """Останавливает обучение, если метрика не улучшается в течение patience эпох.

    При обнаружении нового лучшего значения автоматически сохраняет
    веса модели в checkpoint_path.

    Args:
        patience: число эпох без улучшения до остановки.
        min_delta: минимальное значимое улучшение метрики.
        mode: 'max' — метрика должна расти (Dice), 'min' — убывать (Loss).
        checkpoint_path: путь для сохранения лучших весов модели.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4,
                 mode: str = "max", checkpoint_path: str = "best_model.pth"):
        self.patience        = patience
        self.min_delta       = min_delta
        self.mode            = mode
        self.checkpoint_path = checkpoint_path

        # Инициализируем лучшее значение в зависимости от режима мониторинга
        self.best_value: float = -np.inf if mode == "max" else np.inf
        self.counter:    int   = 0   # счётчик эпох без улучшения
        self.best_epoch: int   = 0   # номер эпохи с лучшим значением
        self.stop:       bool  = False

    def __call__(self, value: float, model: torch.nn.Module, epoch: int) -> bool:
        """Вызывается в конце каждой эпохи с текущим значением метрики.

        Args:
            value: текущее значение отслеживаемой метрики.
            model: модель для сохранения при улучшении.
            epoch: номер текущей эпохи (0-indexed).

        Returns:
            True если нужно остановить обучение, иначе False.
        """
        # Проверяем улучшение с учётом минимального порога min_delta
        improved = (
            (value > self.best_value + self.min_delta)
            if self.mode == "max"
            else (value < self.best_value - self.min_delta)
        )
        if improved:
            self.best_value = value
            self.counter    = 0
            self.best_epoch = epoch
            # Сохраняем только веса (state_dict), не всю модель — экономит место
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"  [+] EarlyStopping: новый рекорд {value:.4f} — чекпоинт сохранён.")
        else:
            self.counter += 1
            print(f"  EarlyStopping: улучшения нет ({self.counter}/{self.patience}).")
            if self.counter >= self.patience:
                self.stop = True
                print(f"  [-] Ранняя остановка. Лучшая эпоха: {self.best_epoch+1}, "
                      f"лучшее значение: {self.best_value:.4f}")
        return self.stop


def plot_history(history: dict, save_path: str = None) -> None:
    """Строит и сохраняет графики Loss и Dice Coefficient по эпохам.

    Отображает два графика:
    - Левый: train/val функция потерь (BCE + Dice)
    - Правый: train/val коэффициент Dice

    Args:
        history: словарь с ключами:
            'train_loss', 'val_loss' — потери по эпохам;
            'train_dice', 'val_dice' — Dice по эпохам.
        save_path: путь для сохранения PNG. По умолчанию PLOT_DIR/training_history.png.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Левый график — функция потерь
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss (BCE + Dice)", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Правый график — коэффициент Dice
    axes[1].plot(epochs, history["train_dice"], "b-o", label="Train Dice", markersize=4)
    axes[1].plot(epochs, history["val_dice"],   "r-o", label="Val Dice",   markersize=4)
    axes[1].set_title("Dice Coefficient", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = save_path or os.path.join(PLOT_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"График обучения сохранён -> {path}")
