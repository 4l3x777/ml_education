import os

# Пути к данным
# DATA_DIR можно переопределить через переменную окружения, по умолчанию ./data
DATA_DIR   = os.environ.get("DATA_DIR", "\\data")
TRAIN_IMG  = os.path.join(DATA_DIR, "train_images")  # папка с тренировочными изображениями
TEST_IMG   = os.path.join(DATA_DIR, "test_images")   # папка с тестовыми изображениями
TRAIN_CSV  = os.path.join(DATA_DIR, "train.csv")     # CSV с RLE-масками
PLOT_DIR   = "\\plots"                                # папка для сохранения графиков
CHECKPOINT = "\\checkpoints\\best_model.pth"          # путь для сохранения лучшей модели

# Создаём директории если их нет
#os.makedirs(PLOT_DIR, exist_ok=True)
#os.makedirs("\\checkpoints", exist_ok=True)

# Классы облачных образований
# 4 класса согласно Understanding Clouds from Satellite Images
CLASSES     = ["Fish", "Flower", "Gravel", "Sugar"]
NUM_CLASSES = len(CLASSES)  # 4

# Параметры модели
ENCODER         = "timm-efficientnet-b8"  # энкодер (backbone) — EfficientNet-B8
ENCODER_WEIGHTS = "imagenet"         # предобученные веса энкодера
IN_CHANNELS     = 3                  # RGB изображения

# Гиперпараметры обучения
IMG_SIZE      = (320, 480)  # (H, W) — размер для ресайза (уменьшаем с 1400x2100)
BATCH_SIZE    = 12          # размер мини-батча; уменьшите до 4 при нехватке VRAM
NUM_EPOCHS    = 50          # максимальное число эпох (early stopping может остановить раньше)
LEARNING_RATE = 3e-4        # начальная скорость обучения для AdamW
WEIGHT_DECAY  = 1e-5        # L2-регуляризация (weight decay) в AdamW
VAL_SPLIT     = 0.15        # доля валидационных изображений (15%)
SEED          = 42          # зерно генератора случайных чисел для воспроизводимости

# Ранняя остановка (Early Stopping)
ES_PATIENCE  = 5     # число эпох без улучшения до остановки обучения
ES_MIN_DELTA = 1e-4  # минимальное значимое улучшение метрики
ES_MODE      = "max" # режим 'max': метрика Dice должна расти

# Планировщик скорости обучения (ReduceLROnPlateau)
SCHED_FACTOR   = 0.5  # множитель уменьшения LR при плато: new_lr = lr * factor
SCHED_PATIENCE = 2    # число эпох без улучшения до уменьшения LR
