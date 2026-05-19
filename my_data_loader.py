"""
Пример реализации загрузчика данных для YOLO Template.

Реализуйте методы collect_image_paths() и iter_split() для вашего датасета.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image

from yolo_template.data_preparation.prepare_yolo_data import BaseDataLoader


class MyDataLoader(BaseDataLoader):
    """
    Загрузчик данных для вашего датасета.

    Реализуйте методы для загрузки ваших данных.

    Example:
        loader = MyDataLoader("./path/to/dataset")
        loader.split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        for item in loader.iter_split("train"):
            image = item["image"]
            annotations = item["annotations"]
            # Обработка...
    """

    def __init__(self, data_dir: str, **kwargs):
        """
        Инициализация загрузчика.

        Args:
            data_dir: Путь к исходному датасету
            **kwargs: Дополнительные параметры
        """
        super().__init__(data_dir, **kwargs)

    def collect_image_paths(self) -> List[Dict[str, Any]]:
        """
        Собрать информацию обо всех изображениях.

        Returns:
            Список словарей с метаданными:
            - "image_path": str - путь к изображению
            - "ann_path": str - путь к аннотациям (опционально)
        """
        # TODO: Реализуйте сбор путей к вашим данным
        paths = []

        # Пример для структуры:
        # dataset/
        # ├── images/
        # │   ├── img1.jpg
        # │   └── img2.jpg
        # └── annotations/
        #     ├── img1.json
        #     └── img2.json

        images_dir = self.data_dir / "images"
        ann_dir = self.data_dir / "annotations"

        if images_dir.exists():
            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path = images_dir / img_file
                    ann_path = ann_dir / img_file.replace(".jpg", ".json").replace(".png", ".json")

                    paths.append({
                        "image_path": str(img_path),
                        "ann_path": str(ann_path) if ann_path.exists() else None,
                    })

        return paths

    def _load_annotations(self, ann_path: str) -> List[Dict[str, Any]]:
        """
        Загрузить аннотации из файла.

        Args:
            ann_path: Путь к файлу аннотаций

        Returns:
            Список аннотаций в формате:
            [
                {
                    "class_id": 0,
                    "bbox": [x_min, y_min, x_max, y_max],  # для детекции
                    "segment": [x1, y1, x2, y2, ...],      # для сегментации (опционально)
                },
                ...
            ]
        """
        # TODO: Реализуйте загрузку аннотаций в вашем формате
        annotations = []

        if not ann_path or not os.path.exists(ann_path):
            return annotations

        # Пример для JSON формата:
        # import json
        # with open(ann_path, "r") as f:
        #     data = json.load(f)
        # for obj in data.get("objects", []):
        #     annotations.append({
        #         "class_id": obj["class_id"],
        #         "bbox": obj["bbox"],  # [x_min, y_min, x_max, y_max]
        #     })

        return annotations

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
        """
        paths = self.get_split_paths(split)

        for item in paths:
            image_path = item["image_path"]
            ann_path = item.get("ann_path")

            # Загружаем изображение
            if cv2 is not None:
                image = cv2.imread(image_path)
            else:
                img = Image.open(image_path).convert("RGB")
                image = np.array(img)[:, :, ::-1]  # RGB -> BGR

            # Загружаем аннотации
            annotations = self._load_annotations(ann_path) if ann_path else []

            yield {
                "image": image,
                "annotations": annotations,
                "metadata": item,
            }


# Для запуска через CLI создайте configs/data_config.json:
"""
{
    "data_dir": "./path/to/your/dataset",
    "output_dir": "./datasets/my_dataset",
    "class_names": ["class1", "class2"],
    "task": "detect",
    "loader_module": "my_data_loader",
    "loader_class": "MyDataLoader",
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15
}
"""
