# YOLO Training Template

Универсальный шаблон для быстрого и удобного запуска обучения моделей YOLO (v8/v9/v10 и новее) на любых пользовательских данных. Включает готовые конфигурации обучения и бенчмаркинга, а также гибкий класс для конвертации разметки в формат YOLO, который можно адаптировать под частные датасеты.

## Основные возможности

- **Автоматическая подготовка данных** – конвертация произвольных форматов аннотаций в стандартный YOLO-формат
- **Гибкий загрузчик данных** – базовый класс для реализации поддержки ваших датасетов
- **Готовые конфигурации** – настройки обучения и оценки в YAML/JSON
- **Бенчмаркинг** – оценка качества модели с метриками детекции и сегментации
- **Минимальная зависимость от инфраструктуры** – работает из коробки с локальным GPU или CPU

## Структура проекта

```
yolo_train_template/
├── configs/                    # Конфигурационные файлы
│   ├── train/                  # Настройки обучения
│   │   └── train_config.yaml
│   ├── benchmark/              # Настройки бенчмаркинга
│   │   └── benchmark.yaml
│   └── dataset/                # Конфигурации датасетов
│       └── data_config.json
├── scripts/                    # Основные скрипты
│   ├── train.py               # Обучение модели
│   └── benchmark.py           # Оценка модели
├── yolo_template/             # Пакет с утилитами
│   ├── data_preparation/      # Подготовка данных
│   │   ├── prepare_yolo_data.py  # Основной класс конвертации
│   │   └── visualize_dataset.py  # Визуализация датасета
│   └── visualization/         # Утилиты визуализации
├── my_data_loader.py          # Пример реализации загрузчика
├── pyproject.toml             # Зависимости и метаданные
├── train.bat                  # Скрипт обучения (Windows)
├── benchmark.bat              # Скрипт оценки (Windows)
└── prepare_data.bat           # Скрипт подготовки данных (Windows)
```

## Быстрый старт

### 1. Установка

```bash
git clone <repository-url>
cd yolo_train_template
pip install -e .
```

### 2. Подготовка данных

#### 2.1 Реализуйте загрузчик данных

Создайте или модифицируйте `my_data_loader.py`, унаследовавшись от `BaseDataLoader`:

```python
from yolo_template.data_preparation.prepare_yolo_data import BaseDataLoader

class MyDataLoader(BaseDataLoader):
    def collect_image_paths(self) -> List[Dict[str, Any]]:
        """Собрать информацию о всех изображениях."""
        # Ваша логика сбора путей к изображениям и аннотациям
        ...

    def iter_split(self, split: str):
        """Итератор по изображениям сплита (train/val/test)."""
        # Загрузка изображений и аннотаций для указанного сплита
        ...
```

Подробный пример доступен в `my_data_loader.py`.

#### 2.2 Формат аннотаций

Загрузчик должен возвращать аннотации в следующем формате:

```python
# Для детекции
{
    "class_id": 0,
    "bbox": [x_min, y_min, x_max, y_max]  # абсолютные координаты в пикселях
}

# Для сегментации
{
    "class_id": 0,
    "segment": [x1, y1, x2, y2, ...]  # полигон, абсолютные координаты
}
```

#### 2.3 Настройка конфигурации датасета

Отредактируйте `configs/dataset/data_config.json`:

```json
{
    "data_dir": "./path/to/your/dataset",
    "output_dir": "./data/my_dataset",
    "class_names": ["class1", "class2"],
    "class_mapping": {},
    "task": "detect",
    "loader_module": "my_data_loader",
    "loader_class": "MyDataLoader",
    "loader_kwargs": {
        "custom_param": "value"
    },
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15
}
```

#### 2.4 Запуск подготовки

```bash
# Windows
prepare_data.bat

# Linux/Mac
python -m yolo_template.data_preparation.prepare_yolo_data --config configs/dataset/data_config.json
```

После выполнения в `output_dir` создается структура в формате YOLO:

```
data/my_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Файл `data.yaml` содержит конфигурацию датасета для Ultralytics YOLO:

```yaml
path: /absolute/path/to/data/my_dataset
train: images/train
val: images/val
test: images/test
names:
  0: class1
  1: class2
```

### 3. Обучение модели

#### 3.1 Настройка конфигурации обучения

Отредактируйте `configs/train/train_config.yaml`:

```yaml
model: yolov8n.pt
data: ./data/my_dataset/data.yaml
epochs: 100
batch: 16
imgsz: 640
```

#### 3.2 Запуск обучения

```bash
# Windows
train.bat configs\train\train_config.yaml

# Linux/Mac
python scripts/train.py --config configs/train/train_config.yaml
```

Результаты сохраняются в `log/<experiment_name>/`:
- `weights/best.pt` – лучшие веса модели
- `results.csv` – метрики по эпохам
- `results.png` – графики обучения

### 4. Бенчмаркинг

#### 4.1 Настройка конфигурации оценки

Отредактируйте `configs/benchmark/benchmark.yaml`:

```yaml
task: detect
model: log/experiment/weights/best.pt
data: ./data/my_dataset/data.yaml
split: val
plots: True
visualize: True
```

#### 4.2 Запуск оценки

```bash
# Windows
benchmark.bat configs\benchmark\benchmark.yaml

# Linux/Mac
python scripts/benchmark.py --config configs/benchmark/benchmark.yaml
```

Метрики оценки:
- **Детекция:** mAP50, mAP50-95, Precision, Recall, F1-score
- **Сегментация:** mask_mAP50, mask_mAP50-95, Dice coefficient

## Требования

- Python >= 3.9
- CUDA-compatible GPU (рекомендуется для обучения)
- Установленные зависимости из `pyproject.toml`

## Автор

Egor K. <yegork.science@gmail.com>

## Лицензия

Проект распространяется под лицензией MIT.
