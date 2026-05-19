"""
Визуализация YOLO-датасета.

Модуль для визуализации аннотаций в формате YOLO.
Создаёт изображения с наложенными bounding box или масками.

Использование:
    python -m yolo_template.data_preparation.visualize_dataset \\
        --data-dir ./data/my_dataset \\
        --output-dir ./data/my_dataset/visualizations \\
        --count 20 \\
        --task detect
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python не установлен. Визуализация невозможна.")


# Цвета для классов (BGR формат)
DEFAULT_COLORS = [
    (255, 0, 0),      # Красный
    (0, 255, 0),      # Зеленый
    (0, 0, 255),      # Синий
    (255, 255, 0),    # Желтый
    (255, 0, 255),    # Пурпурный
    (0, 255, 255),    # Голубой
    (128, 0, 0),      # Темно-красный
    (0, 128, 0),      # Темно-зеленый
    (0, 0, 128),      # Темно-синий
    (128, 128, 0),    # Оливковый
    (128, 0, 128),    # Пурпурный темный
    (0, 128, 128),    # Бирюзовый темный
    (255, 128, 0),    # Оранжевый
    (255, 0, 128),    # Розовый
    (128, 255, 0),    # Лайм
    (0, 255, 128),    # Мятный
]


def load_yolo_label(
    label_path: str,
    image_width: int,
    image_height: int,
    task: str = "detect",
) -> List[Dict]:
    """
    Загрузить аннотации из YOLO-файла.

    Args:
        label_path: Путь к .txt файлу с аннотациями
        image_width: Ширина изображения
        image_height: Высота изображения
        task: Тип задачи ('detect' или 'segment')

    Returns:
        Список аннотаций с bbox и class_id
    """
    annotations = []

    if not os.path.exists(label_path):
        return annotations

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            if task == "segment" and len(parts) >= 7:
                # Сегментация: полигон
                polygon = [float(x) for x in parts[1:]]

                # Конвертируем в абсолютные координаты
                polygon_abs = []
                for i in range(0, len(polygon), 2):
                    x = int(polygon[i] * image_width)
                    y = int(polygon[i + 1] * image_height)
                    polygon_abs.append((x, y))

                annotations.append({
                    "class_id": class_id,
                    "polygon": polygon_abs,
                })
            else:
                # Детекция: bbox
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Конвертируем в абсолютные координаты
                x_min = int((x_center - width / 2) * image_width)
                y_min = int((y_center - height / 2) * image_height)
                x_max = int((x_center + width / 2) * image_width)
                y_max = int((y_center + height / 2) * image_height)

                annotations.append({
                    "class_id": class_id,
                    "bbox": [x_min, y_min, x_max, y_max],
                })

    return annotations


def visualize_image(
    image_path: str,
    label_path: str,
    class_names: Optional[List[str]] = None,
    task: str = "detect",
    show_confidence: bool = False,
) -> Optional[np.ndarray]:
    """
    Визуализировать аннотации на изображении.

    Args:
        image_path: Путь к изображению
        label_path: Путь к YOLO label файлу
        class_names: Список имен классов
        task: Тип задачи ('detect' или 'segment')
        show_confidence: Показывать confidence (если есть)

    Returns:
        Изображение с наложенными аннотациями
    """
    if not CV2_AVAILABLE:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    h, w = image.shape[:2]

    annotations = load_yolo_label(str(label_path), w, h, task)

    for ann in annotations:
        class_id = ann["class_id"]
        color = DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)]

        # Инициализируем координаты для подписи
        x_min, y_min = 0, 0

        if task == "segment" and "polygon" in ann:
            # Рисуем полигон
            polygon = np.array(ann["polygon"], dtype=np.int32)
            cv2.drawContours(image, [polygon], -1, color, 2)

            # Рисуем заполненную маску с прозрачностью
            overlay = image.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            # Вычисляем bbox из полигона для подписи
            x_coords = [p[0] for p in polygon]
            y_coords = [p[1] for p in polygon]
            x_min = min(x_coords)
            y_min = min(y_coords)
        elif "bbox" in ann:
            # Рисуем bbox
            x_min, y_min, x_max, y_max = ann["bbox"]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        else:
            # Нет ни полигона, ни bbox - пропускаем
            continue

        # Подпись класса
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"class_{class_id}"

        # Рисуем фон для текста
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x_min, y_min - text_h - baseline - 3),
            (x_min + text_w, y_min),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x_min, y_min - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return image


def visualize_dataset(
    data_dir: str,
    output_dir: str,
    split: str = "val",
    count: int = 20,
    class_names: Optional[List[str]] = None,
    task: str = "detect",
    seed: int = 42,
) -> None:
    """
    Визуализировать выборку изображений из датасета.

    Args:
        data_dir: Путь к YOLO-датасету (директория с data.yaml)
        output_dir: Директория для сохранения визуализаций
        split: Сплит для визуализации ('train', 'val', 'test')
        count: Количество изображений для визуализации
        class_names: Список имен классов
        task: Тип задачи ('detect' или 'segment')
        seed: Сид для случайной выборки
    """
    if not CV2_AVAILABLE:
        print("Error: opencv-python не установлен. Визуализация невозможна.")
        return

    data_dir = Path(data_dir)
    output_dir = Path(output_dir) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = data_dir / "images" / split
    labels_dir = data_dir / "labels" / split

    if not images_dir.exists():
        print(f"Error: Директория изображений не найдена: {images_dir}")
        return

    # Получаем список изображений
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in os.listdir(images_dir)
        if Path(f).suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"Error: Изображения не найдены в {images_dir}")
        return

    # Выбираем случайные изображения
    random.seed(seed)
    selected = random.sample(image_files, min(count, len(image_files)))

    print(f"Visualizing {len(selected)} images from '{split}' split for task {task}...")

    for img_file in selected:
        img_path = images_dir / img_file
        label_path = labels_dir / Path(img_file).with_suffix(".txt").name

        result = visualize_image(
            str(img_path),
            str(label_path),
            class_names=class_names,
            task=task,
        )

        if result is not None:
            output_path = output_dir / img_file
            cv2.imwrite(str(output_path), result)
            print(f"  Сохранено: {output_path}")

    print(f"\nВизуализации сохранены в: {output_dir}")


def load_class_names_from_yaml(data_dir: str) -> Optional[List[str]]:
    """
    Загрузить имена классов из data.yaml.

    Args:
        data_dir: Путь к YOLO-датасету

    Returns:
        Список имен классов или None
    """
    import yaml

    yaml_path = Path(data_dir) / "data.yaml"
    if not yaml_path.exists():
        return None

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    if isinstance(names, dict):
        # Сортируем по ключам
        names = [names[i] for i in sorted(names.keys())]

    return names


def load_task_from_yaml(data_dir: str) -> str:
    """
    Загрузить тип задачи из data.yaml.

    Args:
        data_dir: Путь к YOLO-датасету

    Returns:
        Тип задачи ('detect' или 'segment') или 'detect' по умолчанию
    """
    import yaml

    yaml_path = Path(data_dir) / "data.yaml"
    if not yaml_path.exists():
        return "detect"

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return data.get("task", "detect")


def main():
    """Точка входа для CLI."""
    parser = argparse.ArgumentParser(
        description="Визуализация YOLO-датасета",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Визуализировать val сплит:
  python -m yolo_template.data_preparation.visualize_dataset \\
      --data-dir ./data/my_dataset \\
      --output-dir ./data/my_dataset/visualizations

  # Визуализировать train сплит с 50 изображениями:
  python -m yolo_template.data_preparation.visualize_dataset \\
      --data-dir ./data/my_dataset \\
      --output-dir ./data/my_dataset/visualizations \\
      --split train \\
      --count 50

  # Для сегментации:
  python -m yolo_template.data_preparation.visualize_dataset \\
      --data-dir ./data/my_dataset \\
      --task segment
        """,
    )

    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        required=True,
        help="Путь к YOLO-датасету (директория с data.yaml)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Директория для сохранения визуализаций "
             "(по умолчанию: <data-dir>/visualizations)",
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Сплит для визуализации (по умолчанию: val)",
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=20,
        help="Количество изображений для визуализации (по умолчанию: 20)",
    )
    parser.add_argument(
        "--class-names", "-n",
        type=str,
        nargs="+",
        default=None,
        help="Список имен классов (если не указаны, будут загружены из data.yaml)",
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        choices=["detect", "segment"],
        default="detect",
        help="Тип задачи: detect или segment (по умолчанию: detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Сид для случайной выборки (по умолчанию: 42)",
    )

    args = parser.parse_args()

    # Определяем output_dir
    if args.output_dir is None:
        args.output_dir = str(Path(args.data_dir) / "visualizations")

    # Загружаем имена классов из data.yaml если не указаны
    class_names = args.class_names
    if class_names is None:
        class_names = load_class_names_from_yaml(args.data_dir)
        if class_names:
            print(f"Class names loaded from data.yaml: {class_names}")
        else:
            print("Warning: Failed to load class names from data.yaml")
            class_names = []

    # Загружаем task из data.yaml если не указан в аргументах
    task = args.task
    if task == "detect":  # Значение по умолчанию - пробуем загрузить из yaml
        loaded_task = load_task_from_yaml(args.data_dir)
        if loaded_task:
            task = loaded_task
            print(f"Task loaded from data.yaml: {task}")

    print("=" * 60)
    print("YOLO Template - Dataset Visualization")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Dataset:     {args.data_dir}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Split:       {args.split}")
    print(f"  Count:       {args.count}")
    print(f"  Classes:     {class_names}")
    print(f"  Task:        {task}")
    print()

    visualize_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        count=args.count,
        class_names=class_names,
        task=task,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
