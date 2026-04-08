import torch
import numpy as np
from config import CLASSES


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> dict:
    """Вычисляет Dice Coefficient для каждого класса и среднее по классам.

    Dice = 2 * |Pred ∩ GT| / (|Pred| + |GT|)
    Значение 1.0 — идеальное совпадение предсказания с разметкой.
    Значение 0.0 — полное несовпадение.

    Args:
        logits   : сырой выход модели (B, C, H, W) без применения sigmoid.
        targets  : ground-truth маски (B, C, H, W), значения 0.0 или 1.0.
        threshold: порог бинаризации вероятностей sigmoid(logits).
        smooth   : сглаживающий член Лапласа для устойчивости при пустых масках.

    Returns:
        Словарь {класс: dice_score, 'mean': mean_dice}.
    """
    # Бинаризуем предсказания: применяем sigmoid, затем порог 0.5
    preds = (torch.sigmoid(logits) >= threshold).float()

    # Разворачиваем пространственные размеры для векторизованного вычисления
    p = preds.view(preds.size(0), preds.size(1), -1)       # (B, C, H*W)
    t = targets.view(targets.size(0), targets.size(1), -1)

    # Суммируем по батчу (dim=0) и по пикселям (dim=2) — получаем вектор длиной C
    intersection = (p * t).sum(dim=(0, 2))               # (C,)
    union        = p.sum(dim=(0, 2)) + t.sum(dim=(0, 2)) # (C,)
    per_class    = (2.0 * intersection + smooth) / (union + smooth)  # (C,)

    result = {c: per_class[i].item() for i, c in enumerate(CLASSES)}
    result["mean"] = per_class.mean().item()  # средний Dice по всем классам
    return result


@torch.no_grad()  # отключаем вычисление градиентов — ускоряет инференс и экономит память
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple:
    """Проводит один полный проход по валидационному набору данных.

    Накапливает логиты всего датасета в памяти CPU, затем вычисляет
    финальные метрики сразу по всему валидационному множеству.

    Args:
        model    : обученная модель U-Net.
        loader   : DataLoader валидационного набора.
        criterion: функция потерь (BCEDiceLoss).
        device   : torch.device — 'cuda' или 'cpu'.

    Returns:
        (avg_loss: float, dice_scores: dict) — средняя потеря и словарь Dice метрик.
    """
    model.eval()  # переключаем в режим инференса:
                  # BatchNorm использует скользящую статистику, Dropout отключён
    total_loss = 0.0
    all_logits, all_targets = [], []

    for images, masks in loader:
        # non_blocking=True позволяет асинхронный перенос данных на GPU
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        #logits = model(images)
        # TTA (4 аугментации)
        tta_logits = []
        tta_logits.append(model(images))
        tta_logits.append(model(torch.flip(images, [3])).flip(3))
        tta_logits.append(model(torch.flip(images, [2])).flip(2))
        tta_logits.append(model(torch.flip(images, [2,3])).flip(2,3))    
        logits = torch.mean(torch.stack(tta_logits), dim=0)

        loss   = criterion(logits, masks)
        # Умножаем на размер батча для последующего нахождения взвешенного среднего
        total_loss += loss.item() * images.size(0)

        # Переносим результаты на CPU для экономии VRAM во время накопления
        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    # Объединяем предсказания и маски всего датасета в единые тензоры
    all_logits  = torch.cat(all_logits,  dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss    = total_loss / len(loader.dataset)           # средняя потеря на один образец
    dice_scores = dice_coefficient(all_logits, all_targets)  # метрики по всему датасету
    return avg_loss, dice_scores
