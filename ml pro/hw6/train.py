#!/usr/bin/env python3
"""Основной скрипт обучения модели сегментации облачных образований.

Архитектура: U-Net с энкодером EfficientNet-B8 (предобученным на ImageNet).
Датасет: Kaggle 'Understanding Clouds from Satellite Images'.
Функция потерь: взвешенная сумма BCE + Soft Dice.
Метрика: Dice Coefficient (per-class и среднее).
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    TRAIN_IMG, TRAIN_CSV, CHECKPOINT,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    VAL_SPLIT, SEED,
    ES_PATIENCE, ES_MIN_DELTA, ES_MODE,
    SCHED_FACTOR, SCHED_PATIENCE,
    CLASSES,
)
from dataset  import CloudDataset
from model    import build_model, BCEDiceLoss
from evaluate import evaluate_model, dice_coefficient
from utils    import set_seed, EarlyStopping, plot_history


def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler=None
) -> tuple:
    """Выполняет одну обучающую эпоху.

    Включает forward pass, вычисление потерь, backward pass с gradient clipping
    и шаг оптимизатора. Поддерживает Automatic Mixed Precision (AMP) для GPU.

    Args:
        model    : U-Net модель.
        loader   : DataLoader тренировочного набора.
        optimizer: AdamW оптимизатор.
        criterion: BCEDiceLoss функция потерь.
        device   : torch.device ('cuda' или 'cpu').
        scaler   : GradScaler для AMP (None если устройство — CPU).

    Returns:
        (avg_loss: float, mean_dice: float) — потеря и Dice за эпоху.
    """
    model.train()  # включаем режим обучения (BatchNorm и Dropout активны)
    total_loss = 0.0
    all_logits, all_targets = [], []

    bar = tqdm(loader, desc="  train", leave=False)
    for images, masks in bar:
        # Переносим батч на устройство (GPU/CPU)
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()  # обнуляем градиенты перед каждым батчем

        if scaler is not None:  # GPU путь с Automatic Mixed Precision
            with torch.amp.autocast():  # вычисления в float16 для ускорения
                logits = model(images)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()        # масштабируем градиенты для float16
            scaler.unscale_(optimizer)           # снимаем масштаб перед клиппингом
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            scaler.step(optimizer)               # шаг оптимизатора (с проверкой overflow)
            scaler.update()                      # обновляем масштаб для следующего шага
        else:  # CPU путь (без AMP)
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            optimizer.step()

        # Накапливаем взвешенную сумму потерь (для вычисления среднего)
        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.detach().cpu())  # детачим от графа и переносим на CPU
        all_targets.append(masks.cpu())
        bar.set_postfix(loss=f"{loss.item():.4f}")  # обновляем прогресс-бар

    # Объединяем результаты всей эпохи для расчёта финальных метрик
    all_logits  = torch.cat(all_logits,  dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_loss    = total_loss / len(loader.dataset)  # средняя потеря на образец
    dice        = dice_coefficient(all_logits, all_targets)["mean"]
    return avg_loss, dice


def main():
    set_seed(SEED)  # фиксируем все генераторы для воспроизводимости
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Подготовка данных
    # Читаем CSV с разметкой; NaN (нет маски) заменяем пустой строкой
    df = pd.read_csv(TRAIN_CSV).fillna("")

    # Получаем уникальные идентификаторы изображений для корректного разбиения:
    # важно делить именно изображения, а не строки CSV — иначе одно фото
    # может попасть и в train, и в val
    df[["image_id", "class"]] = df["Image_Label"].str.rsplit("_", n=1, expand=True)
    image_ids = df["image_id"].unique()

    # Стратифицированное разбиение на train/val
    train_ids, val_ids = train_test_split(
        image_ids, test_size=VAL_SPLIT, random_state=SEED
    )
    train_df = df[df["image_id"].isin(train_ids)].drop(columns=["image_id", "class"])
    val_df   = df[df["image_id"].isin(val_ids)].drop(columns=["image_id", "class"])

    print(f"Train: {len(train_ids)} images | Val: {len(val_ids)} images")

    # Создаём датасеты с соответствующими аугментациями
    train_dataset = CloudDataset(train_df, TRAIN_IMG, phase="train")
    val_dataset   = CloudDataset(val_df,   TRAIN_IMG, phase="val")

    # DataLoader с параллельной загрузкой данных
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0,   # 4 параллельных процесса загрузки
        pin_memory=True, # закрепляем память для быстрого переноса на GPU
        drop_last=True,  # отбрасываем неполный последний батч
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Модель, оптимизатор, планировщик
    model     = build_model().to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)  # 50% BCE + 50% Dice

    # AdamW — улучшенный Adam с корректным L2-регуляризатором
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # ReduceLROnPlateau: уменьшает LR когда val_dice перестаёт улучшаться
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE, verbose=True,
    )

    # GradScaler для Automatic Mixed Precision (ускоряет обучение на GPU)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    early_stop = EarlyStopping(
        patience=ES_PATIENCE,
        min_delta=ES_MIN_DELTA,
        mode=ES_MODE,
        checkpoint_path=CHECKPOINT,
    )

    # Основной цикл обучения
    history = {"train_loss": [], "val_loss": [],
               "train_dice": [], "val_dice": []}

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Тренировочный проход по всему датасету
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        # Валидационный проход (без градиентов)
        val_loss, val_scores   = evaluate_model(
            model, val_loader, criterion, device
        )
        val_dice = val_scores["mean"]

        # Шаг планировщика: обновляем LR на основе val_dice
        scheduler.step(val_dice)

        # Сохраняем метрики для построения кривых обучения
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        elapsed = time.time() - t0
        print(
            f"  Loss  train={train_loss:.4f}  val={val_loss:.4f}\n"
            f"  Dice  train={train_dice:.4f}  val={val_dice:.4f}  "
            f"({elapsed:.0f}s)"
        )
        # Выводим Dice по каждому из 4 классов
        per_class_str = "  ".join(
            f"{c}={val_scores[c]:.3f}" for c in CLASSES
        )
        print(f"  Val per-class Dice: {per_class_str}")

        # Проверяем условие ранней остановки; сохраняем чекпоинт при улучшении
        if early_stop(val_dice, model, epoch):
            break

    # Финальная оценка
    print("\nTraining complete. Loading best checkpoint...")
    # Загружаем веса лучшей эпохи для финального тестирования
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

    final_val_loss, final_scores = evaluate_model(
        model, val_loader, criterion, device
    )
    print(f"\nFinal Val Loss : {final_val_loss:.4f}")
    print(f"Final Val Dice : {final_scores['mean']:.4f}")
    for c in CLASSES:
        print(f"  {c:8s}: {final_scores[c]:.4f}")

    # Строим и сохраняем графики кривых обучения
    plot_history(history)


if __name__ == "__main__":
    main()
