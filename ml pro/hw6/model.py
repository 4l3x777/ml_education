import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from config import ENCODER, ENCODER_WEIGHTS, IN_CHANNELS, NUM_CLASSES


def build_model(encoder: str = ENCODER,
                encoder_weights: str = ENCODER_WEIGHTS) -> nn.Module:
    """Строит U-Net с EfficientNet-B8 в качестве энкодера.

    Архитектура U-Net:
    - Энкодер (EfficientNet-B4): последовательно уменьшает разрешение,
      извлекая иерархические признаки.
    - Декодер: восстанавливает исходное разрешение с помощью up-sampling
      и skip-connections от соответствующих слоёв энкодера.
    - Выход: логиты для каждого из NUM_CLASSES каналов.

    Args:
        encoder: имя энкодера из smp (например, 'efficientnet-b8', 'resnet34').
        encoder_weights: источник предобученных весов ('imagenet' или None).

    Returns:
        nn.Module — модель с выходом (B, NUM_CLASSES, H, W) в виде сырых логитов.
    """
    model = smp.Unet(
        encoder_name    = encoder,
        encoder_weights = encoder_weights,  # предобученные на ImageNet веса энкодера
        in_channels     = IN_CHANNELS,      # 3 канала (RGB)
        classes         = NUM_CLASSES,      # 4 выходных канала (по одному на класс)
        activation      = None,             # без финальной активации — сырые логиты
                                            # sigmoid применяется внутри функции потерь
    )
    return model


class DiceLoss(nn.Module):
    """Soft Dice Loss, усреднённый по классам и батчу.

    В отличие от BCE, Dice Loss инвариантен к классовому дисбалансу
    (важно для задач сегментации, где фоновые пиксели преобладают).
    Использует «мягкую» версию (soft Dice) с sigmoid-вероятностями вместо
    бинарных предсказаний — это делает функцию дифференцируемой.

    Args:
        smooth: сглаживающий член (Лаплас) для предотвращения деления на 0
                и улучшения поведения при пустых масках.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Применяем sigmoid для получения вероятностей в диапазоне [0, 1]
        probs = torch.sigmoid(logits)
        # Разворачиваем пространственные размеры (H*W) для поэлементного вычисления
        p = probs.view(probs.size(0), probs.size(1), -1)    # (B, C, H*W)
        t = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (p * t).sum(dim=2)               # (B, C) — площадь пересечения
        union        = p.sum(dim=2) + t.sum(dim=2)      # (B, C) — сумма площадей
        # Формула Dice с Laplace smoothing:
        # Dice = (2 * |Pred ∩ GT| + smooth) / (|Pred| + |GT| + smooth)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # Возвращаем 1 - Dice: потеря убывает при улучшении предсказания
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Комбинированная функция потерь: взвешенная сумма BCE и Soft Dice.

    BCE (Binary Cross-Entropy) хорошо работает на уровне отдельных пикселей
    и обеспечивает стабильный градиент в начале обучения.
    Dice Loss работает на уровне регионов и устойчив к дисбалансу классов.
    Совместное использование даёт стабильное обучение и точное воспроизведение границ.

    Args:
        bce_weight: вес BCE в суммарной потере (1 - bce_weight идёт на Dice).
        smooth: сглаживающий член в DiceLoss.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        # BCEWithLogitsLoss численно стабильнее, чем sigmoid + BCELoss
        self.bce        = nn.BCEWithLogitsLoss()
        self.dice       = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Взвешенная сумма: 0.5 * BCE + 0.5 * Dice (по умолчанию)
        return (
            self.bce_weight * self.bce(logits, targets)
            + (1 - self.bce_weight) * self.dice(logits, targets)
        )
