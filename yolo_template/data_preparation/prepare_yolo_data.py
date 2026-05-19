"""
Модуль подготовки данных для YOLO моделей.

Предоставляет утилиты для преобразования данных в формат YOLO.
Поддерживает задачи детекции и сегментации.

Архитектура:
    1. Пользователь реализует BaseDataLoader:
       - collect_image_paths() - собрать пути и метаданные
       - iter_split(split) - итератор по сплиту (возвращает image, annotations, metadata)
    
    2. Вызывается prepare_yolo_dataset(loader, ...):
       - loader.split() - разбиение путей на сплиты
       - for split in ["train", "val", "test"]:
           for item in loader.iter_split(split):
               save_to_yolo_format(item, split)

Использование:
    # Через CLI:
    python -m yolo_template.data_preparation.prepare_yolo_data \\
        --data-dir ./my_dataset \\
        --output-dir ./data/my_dataset \\
        --class-names class1 class2 \\
        --loader-module my_data_loader \\
        --loader-class MyDataLoader \\
        --visualize

    # Или через Python API:
    from my_data_loader import MyDataLoader
    from yolo_template.data_preparation import prepare_yolo_dataset
    
    loader = MyDataLoader("./my_dataset")
    prepare_yolo_dataset(
        loader=loader,
        output_dir="./data/my_dataset",
        class_names=["class1", "class2"],
        task="detect",
    )
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python не установлен. Используем PIL для загрузки изображений.")
    try:
        from PIL import Image
        PIL_AVAILABLE = True
    except ImportError:
        PIL_AVAILABLE = False


# =============================================================================
# Базовый класс для загрузчика пользовательских данных
# =============================================================================


class BaseDataLoader:
    """
    Базовый класс для загрузки пользовательских данных.
    
    Принцип работы:
        1. Пользователь реализует collect_image_paths() - возвращает список путей
        2. Вызывается split() - пути разбиваются на сплиты (train/val/test)
        3. Пользователь реализует iter_split(split) - итератор по изображениям сплита
        4. Итератор возвращает: {"image": np.ndarray, "annotations": [...], "metadata": {...}}
    
    Example:
        class MyDataLoader(BaseDataLoader):
            def collect_image_paths(self) -> List[Dict[str, Any]]:
                # Собрать информацию о всех изображениях
                paths = []
                for img_file in os.listdir(self.data_dir / "images"):
                    paths.append({
                        "image_path": str(self.data_dir / "images" / img_file),
                        "ann_path": str(self.data_dir / "labels" / img_file.replace(".jpg", ".txt")),
                    })
                return paths
            
            def iter_split(self, split: str):
                # Итератор по конкретному сплиту
                paths = getattr(self, f"{split}_paths")
                for item in paths:
                    image = cv2.imread(item["image_path"])
                    annotations = self._load_annotations(item["ann_path"])
                    yield {
                        "image": image,
                        "annotations": annotations,
                        "metadata": item,
                    }
    """
    
    def __init__(self, data_dir: str, **kwargs):
        """
        Инициализация загрузчика.
        
        Args:
            data_dir: Путь к исходному датасету
            **kwargs: Дополнительные параметры (crop_size, overlap, и т.д.)
        """
        self.data_dir = Path(data_dir)
        self.kwargs = kwargs
        
        # Атрибуты для путей по сплитам (заполняются после split())
        self.train_paths: Optional[List[Dict[str, Any]]] = None
        self.val_paths: Optional[List[Dict[str, Any]]] = None
        self.test_paths: Optional[List[Dict[str, Any]]] = None
    
    def collect_image_paths(self) -> List[Dict[str, Any]]:
        """
        Собрать информацию обо всех изображениях.
        
        Returns:
            Список словарей с метаданными. Минимальный набор:
            - "source_path": str - уникальный идентификатор (путь к файлу)
            
            Дополнительно можно указать:
            - "ann_path": str - путь к аннотациям
            - Любые другие нужные данные
        """
        raise NotImplementedError("Реализуйте метод collect_image_paths()")
    
    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        """
        Разбить пути на train/val/test сплиты.
        
        Args:
            train_ratio: Доля тренировочной выборки
            val_ratio: Доля валидационной выборки
            test_ratio: Доля тестовой выборки
            seed: Сид для воспроизводимости
        """
        all_paths = self.collect_image_paths()
        
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError(
                f"Сумма ratios должна быть 1.0, получено: {train_ratio + val_ratio + test_ratio}"
            )
        
        random.seed(seed)
        paths = all_paths.copy()
        random.shuffle(paths)
        
        n_total = len(paths)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        self.train_paths = paths[:n_train]
        self.val_paths = paths[n_train : n_train + n_val]
        self.test_paths = paths[n_train + n_val :]
        
        print(f"Разбиение на сплиты:")
        print(f"  Train: {len(self.train_paths)} изображений")
        print(f"  Val:   {len(self.val_paths)} изображений")
        print(f"  Test:  {len(self.test_paths)} изображений")
    
    def iter_split(self, split: str):
        """
        Итератор по изображениям конкретного сплита.
        
        Args:
            split: Название сплита ("train", "val", "test")
        
        Yields:
            Словарь с данными:
            - "image": np.ndarray - изображение (BGR формат)
            - "annotations": List[Dict] - список аннотаций
            - "metadata": Dict - дополнительные метаданные
        
        Пример формата аннотаций для детекции:
            {
                "class_id": 0,  # или "class_name": "A" (для маппинга)
                "bbox": [x_min, y_min, x_max, y_max],  # абсолютные координаты
            }
        
        Пример для сегментации:
            {
                "class_id": 0,
                "segment": [x1, y1, x2, y2, ...],  # полигон, абсолютные координаты
                "bbox": [x_min, y_min, x_max, y_max],  # опционально
            }
        """
        raise NotImplementedError("Реализуйте метод iter_split(split)")
    
    def get_split_paths(self, split: str) -> List[Dict[str, Any]]:
        """
        Получить пути для конкретного сплита.
        
        Args:
            split: Название сплита ("train", "val", "test")
        
        Returns:
            Список путей для сплита
        """
        if split == "train":
            return self.train_paths
        elif split == "val":
            return self.val_paths
        elif split == "test":
            return self.test_paths
        else:
            raise ValueError(f"Неизвестный сплит: {split}")
    
    def __len__(self) -> int:
        """Общее количество изображений."""
        total = 0
        if self.train_paths:
            total += len(self.train_paths)
        if self.val_paths:
            total += len(self.val_paths)
        if self.test_paths:
            total += len(self.test_paths)
        return total


# =============================================================================
# Функции для сохранения данных в формате YOLO
# =============================================================================


def _get_image_size(image: np.ndarray) -> Tuple[int, int]:
    """
    Получить размеры изображения.
    
    Args:
        image: Изображение как numpy массив
        
    Returns:
        (width, height)
    """
    h, w = image.shape[:2]
    return w, h


def _bbox_to_yolo(
    bbox: List[float],
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """
    Конвертировать bounding box из абсолютных координат в формат YOLO.

    Args:
        bbox: [x_min, y_min, x_max, y_max] в пикселях
        image_width: Ширина изображения
        image_height: Высота изображения

    Returns:
        (x_center, y_center, width, height) в нормализованных координатах [0, 1]
    """
    x_min, y_min, x_max, y_max = bbox

    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    # Clip to [0, 1]
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    width = np.clip(width, 0, 1)
    height = np.clip(height, 0, 1)

    return x_center, y_center, width, height


def _polygon_to_yolo(
    polygon: List[float],
    image_width: int,
    image_height: int,
) -> List[float]:
    """
    Конвертировать полигон из абсолютных координат в нормализованный формат YOLO.

    Формат полигона: [x1, y1, x2, y2, x3, y3, ...]

    Args:
        polygon: Список координат полигона [x1, y1, x2, y2, ...]
        image_width: Ширина изображения
        image_height: Высота изображения

    Returns:
        Список нормализованных координат [x1_norm, y1_norm, x2_norm, y2_norm, ...]
    """
    normalized = []

    for i in range(0, len(polygon), 2):
        x = polygon[i] / image_width
        y = polygon[i + 1] / image_height

        # Clip to [0, 1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)

        normalized.extend([x, y])

    return normalized


def _save_yolo_label(
    annotations: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    label_path: str,
    task: str = "detect",
    class_mapping: Optional[Dict[str, str]] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Сохранить аннотации в формате YOLO.

    Формат для детекции:
        <class-index> <x_center> <y_center> <width> <height>

    Формат для сегментации:
        <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>

    Args:
        annotations: Список аннотаций
        image_width: Ширина изображения
        image_height: Высота изображения
        label_path: Путь для сохранения label файла
        task: Тип задачи ('detect' или 'segment')
        class_mapping: Маппинг имен классов {old_name: new_name}
        class_names: Список финальных имен классов (для определения индексов)
    """
    lines = []

    # Создаем обратный маппинг: new_name -> index
    if class_names:
        name_to_idx = {name: i for i, name in enumerate(class_names)}
    else:
        name_to_idx = {}

    for ann in annotations:
        # Получаем имя класса (если есть) и мапим в индекс
        class_name = ann.get("class_name")
        class_id = ann.get("class_id")
        
        # Если есть class_name, применяем маппинг
        if class_name is not None and class_mapping:
            class_name = class_mapping.get(class_name, class_name)
        
        # Если есть class_names, определяем индекс по имени
        if class_name is not None and name_to_idx:
            class_id = name_to_idx.get(class_name, class_id)
        
        # Если class_id все еще None, пропускаем
        if class_id is None:
            continue

        if task == "segment":
            # Сегментация: используем полигоны
            # Проверяем оба ключа: 'segment' или 'polygon'
            polygon = ann.get("segment") or ann.get("polygon")
            
            if polygon is not None:
                # Проверяем минимальное количество точек (нужно минимум 3)
                if len(polygon) < 6:  # 3 точки = 6 координат
                    print(f"  Warning: Полигон имеет меньше 3 точек, пропускается")
                    continue

                # Конвертируем в нормализованные координаты
                normalized_polygon = _polygon_to_yolo(polygon, image_width, image_height)

                # Форматируем строку
                coords_str = " ".join([f"{c:.6f}" for c in normalized_polygon])
                line = f"{class_id} {coords_str}"
                lines.append(line)

            elif "bbox" in ann:
                # Fallback: если нет полигона, используем bbox
                bbox = ann["bbox"]
                x_center, y_center, width, height = _bbox_to_yolo(
                    bbox, image_width, image_height
                )
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                lines.append(line)
        else:
            # Детекция: используем bbox
            if "bbox" in ann:
                bbox = ann["bbox"]
                x_center, y_center, width, height = _bbox_to_yolo(
                    bbox, image_width, image_height
                )
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                lines.append(line)

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def save_yolo_format(
    image: np.ndarray,
    annotations: List[Dict[str, Any]],
    output_image_path: str,
    output_label_path: str,
    task: str = "detect",
    class_mapping: Optional[Dict[str, str]] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Сохранить изображение и аннотации в формате YOLO.

    Args:
        image: Изображение (BGR формат)
        annotations: Список аннотаций
        output_image_path: Путь для сохранения изображения
        output_label_path: Путь для сохранения label файла
        task: Тип задачи ('detect' или 'segment')
        class_mapping: Маппинг имен классов
        class_names: Список финальных имен классов
    """
    # Сохраняем изображение
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image_path), image)

    # Сохраняем аннотации
    w, h = _get_image_size(image)
    
    output_label_path = Path(output_label_path)
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    
    _save_yolo_label(
        annotations, w, h, str(output_label_path), task,
        class_mapping=class_mapping, class_names=class_names
    )


def _create_data_yaml(
    output_dir: str,
    class_names: List[str],
    task: str = "detect",
) -> None:
    """
    Создать data.yaml файл для YOLO.

    Формат согласно документации Ultralytics:
    https://docs.ultralytics.com/modes/train/

    Args:
        output_dir: Корневая директория датасета
        class_names: Список названий классов
        task: Тип задачи ('detect' или 'segment')
    """
    # Формируем словарь с названиями классов
    names_dict = {i: name for i, name in enumerate(class_names)}

    data = {
        "path": str(Path(output_dir).absolute()),  # dataset root dir
        "train": "images/train",  # train images (relative to 'path')
        "val": "images/val",  # val images (relative to 'path')
        "test": "images/test",  # test images (optional)
        "names": names_dict,
        "task": task,  # Сохраняем тип задачи для визуализации
    }

    yaml_path = Path(output_dir) / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"Создан файл {yaml_path}")


def prepare_yolo_dataset(
    loader: BaseDataLoader,
    output_dir: str,
    class_names: List[str],
    class_mapping: Optional[Dict[str, str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    task: str = "detect",
    seed: int = 42,
) -> None:
    """
    Подготовить датасет в формате YOLO.

    Эта функция выполняет полный пайплайн подготовки данных:
    1. Разбивает пути на train/val/test (loader.split())
    2. Для каждого сплита проходит итератором (loader.iter_split())
    3. Сохраняет изображения и аннотации в формате YOLO
    4. Создает data.yaml файл

    Args:
        loader: Загрузчик данных (наследуется от BaseDataLoader)
        output_dir: Директория для сохранения датасета
        class_names: Список финальных названий классов (порядок определяет ID)
        class_mapping: Маппинг имен классов {old_name: new_name}
        train_ratio: Доля тренировочной выборки
        val_ratio: Доля валидационной выборки
        test_ratio: Доля тестовой выборки
        task: Тип задачи ('detect' или 'segment')
        seed: Сид для воспроизводимости

    Example:
        from my_data_loader import MyDataLoader
        from yolo_template.data_preparation import prepare_yolo_dataset
        
        loader = MyDataLoader("./my_dataset")
        prepare_yolo_dataset(
            loader=loader,
            output_dir="./data/my_dataset",
            class_names=["defect_type1", "defect_type2"],
            class_mapping={"A": "defect_type1", "B": "defect_type2"},
            task="detect",
        )
    """
    print(f"Подготовка датасета в {output_dir}...")
    print(f"Классы: {class_names}")
    print(f"Маппинг: {class_mapping}")
    print(f"Задача: {task}")

    # 1. Разбиваем пути на сплиты
    print("\n[1/3] Разбиение на сплиты...")
    loader.split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # 2. Проходим по каждому сплиту и сохраняем данные
    print("\n[2/3] Сохранение данных...")
    
    for split in ["train", "val", "test"]:
        split_paths = loader.get_split_paths(split)
        if not split_paths:
            continue
        
        print(f"\n  Обработка сплита '{split}' ({len(split_paths)} изображений)...")
        
        images_dir = Path(output_dir) / "images" / split
        labels_dir = Path(output_dir) / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, item in enumerate(loader.iter_split(split)):
            image = item["image"]
            annotations = item["annotations"]
            
            # Пропускаем пустые изображения
            if image is None:
                print(f"    Warning: Пустое изображение, пропускается")
                continue
            
            # Сохраняем в формате YOLO
            new_image_name = f"image_{idx:05d}.jpg"
            new_label_name = f"image_{idx:05d}.txt"
            
            save_yolo_format(
                image=image,
                annotations=annotations,
                output_image_path=str(images_dir / new_image_name),
                output_label_path=str(labels_dir / new_label_name),
                task=task,
                class_mapping=class_mapping,
                class_names=class_names,
            )
            
            if (idx + 1) % 100 == 0:
                print(f"    Обработано {idx + 1}/{len(split_paths)}...")

    # 3. Создаем data.yaml
    print("\n[3/3] Создание data.yaml...")
    _create_data_yaml(output_dir, class_names, task)

    print("\n Датасет успешно подготовлен!")
    print(f"  Путь: {Path(output_dir).absolute()}")


# =============================================================================
# CLI интерфейс
# =============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загрузить конфигурацию из JSON файла.
    
    Args:
        config_path: Путь к JSON конфигу
        
    Returns:
        Словарь с конфигурацией
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Точка входа для CLI."""
    parser = argparse.ArgumentParser(
        description="Подготовка данных для YOLO моделей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Использование:
    python -m yolo_template.data_preparation.prepare_yolo_data --config configs/data/data_config.json

Формат config.json:
  {
      "data_dir": "./dataset",
      "output_dir": "./data/wire_dataset",
      "class_names": ["6W", "10W", "text_stamp", "wire"],
      "class_mapping": {"X_defect": "6W", "C_defect": "10W"},
      "task": "segment",
      "loader_module": "wire_data_loader",
      "loader_class": "WireDataLoader",
      "loader_kwargs": {
          "use_crops": false,
          "crop_size": 640,
          "multiple_crops": 1,
          "jitter": 0.1
      },
      "train_ratio": 0.7,
      "val_ratio": 0.15,
      "test_ratio": 0.15,
      "seed": 42,
      "visualize": true,
      "visualize_count": 20
  }
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Путь к JSON конфигурационному файлу",
    )

    args = parser.parse_args()

    # Загружаем конфигурацию
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Извлекаем параметры из конфига
    data_dir = config.get("data_dir")
    output_dir = config.get("output_dir", "./data/dataset")
    class_names = config.get("class_names")
    class_mapping = config.get("class_mapping")
    task = config.get("task", "detect")
    train_ratio = config.get("train_ratio", 0.7)
    val_ratio = config.get("val_ratio", 0.15)
    test_ratio = config.get("test_ratio", 0.15)
    seed = config.get("seed", 42)
    loader_module = config.get("loader_module")
    loader_class = config.get("loader_class")
    loader_kwargs = config.get("loader_kwargs", {})
    visualize = config.get("visualize", False)
    visualize_count = config.get("visualize_count", 20)

    # Проверяем обязательные параметры
    if not data_dir:
        print("Error: 'data_dir' is required in config")
        sys.exit(1)
    if not class_names:
        print("Error: 'class_names' is required in config")
        sys.exit(1)
    if not loader_module or not loader_class:
        print("Error: 'loader_module' and 'loader_class' are required in config")
        sys.exit(1)

    print("=" * 60)
    print("YOLO Template - Подготовка датасета")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Путь до исходных данных:       {data_dir}")
    print(f"  Путь сохранения:     {output_dir}")
    print(f"  Классы:        {class_names}")
    print(f"  Маппинг классов:  {class_mapping}")
    print(f"  Задача:           {task}")
    print(f"  Класс загрузки:         {loader_module}.{loader_class}")
    if loader_kwargs:
        print(f"  Параметры загрузчика:  {loader_kwargs}")
    print(f"  Сохранить визуализации:      {visualize}")
    if visualize:
        print(f"  Число визуализаций: {visualize_count}")
    print(f"  Разбиения выборок:         train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # Импортируем пользовательский загрузчик
    print(f"\nИмпорт загрузчика: {loader_module}.{loader_class}")
    try:
        module = __import__(loader_module, fromlist=[loader_class])
        LoaderClass = getattr(module, loader_class)

        # Передаем параметры в загрузчик через **kwargs
        loader = LoaderClass(data_dir, **loader_kwargs)
    except (ImportError, AttributeError) as e:
        print(f"Loader import error: {e}")
        sys.exit(1)

    # Запускаем подготовку
    prepare_yolo_dataset(
        loader=loader,
        output_dir=output_dir,
        class_names=class_names,
        class_mapping=class_mapping,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        task=task,
        seed=seed,
    )

    # Визуализация если запрошено
    if visualize:
        print("\n" + "=" * 60)
        print("Визуализация набора данных (val)...")
        print("=" * 60)
        try:
            from .visualize_dataset import (
                visualize_dataset,
            )
            visualize_dataset(
                data_dir=output_dir,
                output_dir=str(Path(output_dir) / "visualizations"),
                split="val",
                count=visualize_count,
                class_names=class_names,
                task=task,
                seed=seed,
            )
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")


if __name__ == "__main__":
    main()
