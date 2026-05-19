"""
Визуализация предсказаний YOLO моделей.

Функции и классы для сохранения изображений с наложенными предсказаниями.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Цвета для разных классов (BGR формат для OpenCV)
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


class PredictionVisualizer:
    """
    Класс для визуализации предсказаний YOLO моделей.

    Поддерживает как детекцию (bounding boxes), так и сегментацию (маски).

    Example:
        visualizer = PredictionVisualizer(class_names=["defect", "background"])

        # Для детекции
        visualizer.visualize_detection(
            image=image,
            predictions=predictions,
            save_path="output/detection.png"
        )

        # Для сегментации
        visualizer.visualize_segmentation(
            image=image,
            predictions=predictions,
            save_path="output/segmentation.png"
        )
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        box_thickness: int = 2,
        font_scale: float = 0.5,
        alpha: float = 0.5,
    ):
        """
        Инициализация визуализатора.

        Args:
            class_names: Список названий классов
            colors: Список цветов для классов (BGR формат)
            box_thickness: Толщина рамок bounding box
            font_scale: Масштаб шрифта для подписей
            alpha: Прозрачность для масок сегментации
        """
        self.class_names = class_names or []
        self.colors = colors or DEFAULT_COLORS
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.alpha = alpha

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Получить цвет для класса."""
        return self.colors[class_id % len(self.colors)]

    def _get_class_name(self, class_id: int) -> str:
        """Получить название класса."""
        if class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def visualize_detection(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> np.ndarray:
        """
        Визуализировать предсказания детекции на изображении.

        Args:
            image: Исходное изображение (BGR формат)
            boxes: Bounding boxes [N, 4] в формате [x1, y1, x2, y2]
            scores: Confidence scores [N]
            labels: Class labels [N]
            save_path: Путь для сохранения (если None, не сохранять)
            show: Показывать изображение (требует GUI)

        Returns:
            Изображение с наложенными bounding boxes
        """
        if not CV2_AVAILABLE:
            raise ImportError("opencv-python требуется для визуализации")

        result = image.copy()
        n_boxes = len(boxes)

        for i in range(n_boxes):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box)

            # Получаем класс и цвет
            class_id = int(labels[i]) if labels is not None else 0
            color = self._get_color(class_id)

            # Рисуем bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, self.box_thickness)

            # Добавляем подпись
            if scores is not None:
                score = scores[i]
                label = self._get_class_name(class_id)
                text = f"{label}: {score:.2f}"
            else:
                text = self._get_class_name(class_id)

            # Вычисляем размер текста
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )

            # Рисуем фон для текста
            cv2.rectangle(
                result,
                (x1, y1 - text_h - baseline - 3),
                (x1 + text_w, y1),
                color,
                -1,
            )

            # Рисуем текст
            cv2.putText(
                result,
                text,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
            )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), result)

        if show:
            cv2.imshow("Detection", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result

    def visualize_segmentation(
        self,
        image: np.ndarray,
        masks: Union[np.ndarray, List[np.ndarray]],
        boxes: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> np.ndarray:
        """
        Визуализировать предсказания сегментации на изображении.

        Args:
            image: Исходное изображение (BGR формат)
            masks: Маски [N, H, W] или List[np.ndarray]
            boxes: Bounding boxes [N, 4] (опционально)
            scores: Confidence scores [N]
            labels: Class labels [N]
            save_path: Путь для сохранения
            show: Показывать изображение

        Returns:
            Изображение с наложенными масками
        """
        if not CV2_AVAILABLE:
            raise ImportError("opencv-python требуется для визуализации")

        result = image.copy()
        n_masks = len(masks) if isinstance(masks, list) else masks.shape[0]

        # Рисуем каждую маску отдельно с её цветом
        for i in range(n_masks):
            mask = masks[i] if isinstance(masks, list) else masks[i]
            mask = (mask > 0.5).astype(np.uint8)

            if mask.sum() == 0:
                continue

            class_id = int(labels[i]) if labels is not None else 0
            color = self._get_color(class_id)

            # Рисуем контур маски
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, 2)

            # Заполненная маска с прозрачностью
            overlay = result.copy()
            cv2.fillPoly(overlay, contours, color)
            cv2.addWeighted(overlay, self.alpha, result, 1.0 - self.alpha, 0, result)

            # Подпись класса
            if len(contours) > 0:
                # Находим верхнюю точку маски для подписи
                x_min = float('inf')
                y_min = float('inf')
                for cnt in contours:
                    for point in cnt:
                        x, y = point[0]
                        if y < y_min or (y == y_min and x < x_min):
                            x_min, y_min = x, y

                if labels is not None:
                    label = self._get_class_name(class_id)
                    if scores is not None and i < len(scores):
                        text = f"{label}: {scores[i]:.2f}"
                    else:
                        text = label

                    # Рисуем фон для текста
                    (text_w, text_h), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                    )
                    cv2.rectangle(
                        result,
                        (x_min, y_min - text_h - baseline - 3),
                        (x_min + text_w, y_min),
                        color,
                        -1,
                    )
                    cv2.putText(
                        result,
                        text,
                        (x_min, y_min - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        (255, 255, 255),
                        1,
                    )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), result)

        if show:
            cv2.imshow("Segmentation", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result


def visualize_predictions(
    image: np.ndarray,
    result: Any,
    save_path: str,
    task: str = "detect",
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Визуализировать предсказания модели на изображении.

    Args:
        image: Исходное изображение (BGR формат)
        result: Результат инференса модели (ultralytics result)
        save_path: Путь для сохранения
        task: Тип задачи ('detect' или 'segment')
        class_names: Названия классов
    """
    visualizer = PredictionVisualizer(class_names=class_names or [])

    if task == "segment" and hasattr(result, "masks") and result.masks is not None:
        # Сегментация: рисуем только маски (без bbox)
        masks = result.masks.data.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy() if result.boxes is not None else None
        visualizer.visualize_segmentation(
            image=image,
            masks=masks,
            boxes=None,  # Не рисуем bbox для сегментации
            scores=None,
            labels=labels,  # Передаем labels для цветов
            save_path=save_path,
        )
    else:
        # Детекция
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
        labels = result.boxes.cls.cpu().numpy() if result.boxes is not None else np.array([])

        visualizer.visualize_detection(
            image=image,
            boxes=boxes,
            scores=scores,
            labels=labels,
            save_path=save_path,
        )


def visualize_comparison(
    image: np.ndarray,
    result: Any,
    ground_truth_path: str,
    save_path: str,
    class_names: Optional[List[str]] = None,
    task: str = "detect",
) -> None:
    """
    Визуализировать сравнение предсказания модели с ground truth.

    Args:
        image: Исходное изображение (BGR формат)
        result: Результат инференса модели (ultralytics result)
        ground_truth_path: Путь к YOLO label файлу
        save_path: Путь для сохранения
        class_names: Названия классов
        task: Тип задачи ('detect' или 'segment')
    """
    if not MATPLOTLIB_AVAILABLE or not CV2_AVAILABLE:
        raise ImportError("Требуется matplotlib и opencv-python")

    visualizer = PredictionVisualizer(class_names=class_names or [])

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Создаем фигуру с тремя панелями
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Предсказание
    # Рисуем предсказание в зависимости от задачи
    if task == "segment" and hasattr(result, "masks") and result.masks is not None:
        # Сегментация: рисуем только маски (без bbox)
        masks = result.masks.data.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy() if result.boxes is not None else None
        pred_img = visualizer.visualize_segmentation(
            image=image,
            masks=masks,
            boxes=None,  # Не рисуем bbox для сегментации
            scores=None,
            labels=labels,  # Передаем labels для цветов
        )
    else:
        # Детекция: рисуем bbox
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
        labels = result.boxes.cls.cpu().numpy() if result.boxes is not None else np.array([])
        pred_img = visualizer.visualize_detection(
            image=image,
            boxes=boxes,
            scores=scores,
            labels=labels,
        )

    axes[0].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    # 2. Ground Truth - загружаем из YOLO формата
    gt_img = image.copy()
    if ground_truth_path and Path(ground_truth_path).exists():
        with open(ground_truth_path, "r") as f:
            lines = f.readlines()

        gt_boxes = []
        gt_labels = []
        gt_masks = [] if task == "segment" else None

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            if task == "segment" and len(parts) >= 7:
                # Сегментация: полигон в нормализованных координатах
                polygon = [float(x) for x in parts[1:]]
                
                # Конвертируем в абсолютные координаты
                polygon_abs = []
                for i in range(0, len(polygon), 2):
                    x = int(polygon[i] * w)
                    y = int(polygon[i + 1] * h)
                    polygon_abs.append([x, y])

                gt_masks.append(np.array(polygon_abs, dtype=np.int32))
                gt_labels.append(class_id)
            else:
                # Детекция: bbox в нормализованных координатах
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Конвертируем в абсолютные координаты
                x_min = int((x_center - width / 2) * w)
                y_min = int((y_center - height / 2) * h)
                x_max = int((x_center + width / 2) * w)
                y_max = int((y_center + height / 2) * h)

                gt_boxes.append([x_min, y_min, x_max, y_max])
                gt_labels.append(class_id)

        # Рисуем GT аннотации
        if task == "segment" and gt_masks:
            # Сегментация: рисуем только маски (без bbox)
            for i, (mask, label) in enumerate(zip(gt_masks, gt_labels)):
                color = DEFAULT_COLORS[int(label) % len(DEFAULT_COLORS)]
                cv2.drawContours(gt_img, [mask], -1, color, 2)
                # Заполненная маска с прозрачностью
                overlay = gt_img.copy()
                cv2.fillPoly(overlay, [mask], color)
                cv2.addWeighted(overlay, 0.3, gt_img, 0.7, 0, gt_img)

                # Подпись
                x_min, y_min = mask[:, 0].min(), mask[:, 1].min()
                label_text = class_names[int(label)] if class_names and int(label) < len(class_names) else f"class_{label}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    gt_img,
                    (x_min, y_min - text_h - baseline - 3),
                    (x_min + text_w, y_min),
                    color,
                    -1,
                )
                cv2.putText(
                    gt_img,
                    label_text,
                    (x_min, y_min - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
        elif gt_boxes:
            # Детекция: рисуем bbox
            gt_img = visualizer.visualize_detection(
                gt_img,
                np.array(gt_boxes),
                labels=np.array(gt_labels),
            )

    axes[1].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # 3. Исходное изображение
    axes[2].imshow(image_rgb)
    axes[2].set_title("Original")
    axes[2].axis("off")

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
