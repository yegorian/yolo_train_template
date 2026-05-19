"""
YOLO Template - Универсальный шаблон для обучения и оценки YOLO моделей.

Поддерживает задачи детекции и сегментации с использованием Ultralytics.

Для бенчмаркинга используется стандартная валидация Ultralytics (model.val()).
"""

from yolo_template.visualization import PredictionVisualizer

__all__ = [
    "PredictionVisualizer",
]
