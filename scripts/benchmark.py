"""
Скрипт для бенчмаркинга YOLO моделей.

Использует стандартную валидацию Ultralytics (model.val()).

Использование:
    python scripts/benchmark.py --config configs/benchmark/benchmark.yaml

Аргументы:
    --config: Путь к YAML конфигурационному файлу
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
import os
import random
import cv2
import yaml

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo_template.visualization.prediction_visualizer import (
    visualize_predictions,
    visualize_comparison,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загрузить конфигурацию из YAML файла.

    Args:
        config_path: Путь к конфигурационному файлу

    Returns:
        Словарь с конфигурацией
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def run_benchmark(
    model: Any,
    data_path: str,
    task: str = "detect",
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.25,
    iou: float = 0.45,
    split: str = "test",
    device: Optional[str] = None,
    plots: bool = True,
    save_dir: Optional[str] = None,
    save_json: bool = False,
    amp: bool = True,
) -> Dict[str, float]:
    """
    Запустить бенчмаркинг через model.val() (Ultralytics).

    Args:
        model: YOLO модель
        data_path: Путь к датасету
        task: Тип задачи ('detect' или 'segment')
        imgsz: Размер изображения
        batch: Размер батча
        conf: Порог уверенности
        iou: Порог IoU для NMS
        split: Сплит для оценки
        device: Устройство для вычислений
        plots: Сохранять графики метрик
        save_dir: Директория для сохранения результатов валидации
        save_json: Сохранять результаты в JSON
        amp: Automatic Mixed Precision

    Returns:
        Словарь с метриками
    """
    print(f"\nВалидация (Ultralytics) на сплите '{split}'...")

    # Запускаем валидацию через Ultralytics
    val_kwargs = {
        "data": data_path,
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "conf": conf,
        "iou": iou,
        "device": device,
        "plots": plots,
        "save_json": save_json,
        "amp": amp,
        "verbose": True,
    }

    # Если указана save_dir, сохраняем результаты туда
    if save_dir:
        val_kwargs["project"] = str(Path(save_dir).parent)
        val_kwargs["name"] = str(Path(save_dir).name)

    metrics = model.val(**val_kwargs)

    # Извлекаем метрики
    results = {}

    if task == "detect":
        # Метрики для детекции
        results["mAP50"] = float(metrics.box.map50)
        results["mAP50-95"] = float(metrics.box.map)
        results["precision"] = float(metrics.box.p.mean())
        results["recall"] = float(metrics.box.r.mean())

        # Вычисляем F1-score
        if results["precision"] + results["recall"] > 0:
            results["f1"] = 2 * results["precision"] * results["recall"] / (
                results["precision"] + results["recall"]
            )
        else:
            results["f1"] = 0.0

        # Метрики по каждому классу
        num_classes = len(metrics.box.p)
        for i in range(num_classes):
            class_name = f"class_{i}"
            if hasattr(metrics, 'names') and metrics.names:
                class_name = metrics.names[i] if isinstance(metrics.names, list) else metrics.names.get(i, f"class_{i}")
            
            results[f"{class_name}/precision"] = float(metrics.box.p[i])
            results[f"{class_name}/recall"] = float(metrics.box.r[i])
            results[f"{class_name}/mAP50"] = float(metrics.box.ap50[i])
            results[f"{class_name}/mAP50-95"] = float(metrics.box.ap[i])
            
            # F1-score для класса
            p = results[f"{class_name}/precision"]
            r = results[f"{class_name}/recall"]
            if p + r > 0:
                results[f"{class_name}/f1"] = 2 * p * r / (p + r)
            else:
                results[f"{class_name}/f1"] = 0.0

    elif task == "segment":
        # Метрики для сегментации (bbox)
        results["mAP50"] = float(metrics.box.map50)
        results["mAP50-95"] = float(metrics.box.map)
        results["precision"] = float(metrics.box.p.mean())
        results["recall"] = float(metrics.box.r.mean())

        # Вычисляем F1-score для bbox
        if results["precision"] + results["recall"] > 0:
            results["f1"] = 2 * results["precision"] * results["recall"] / (
                results["precision"] + results["recall"]
            )
        else:
            results["f1"] = 0.0

        # Метрики bbox по каждому классу
        num_classes = len(metrics.box.p)
        for i in range(num_classes):
            class_name = f"class_{i}"
            if hasattr(metrics, 'names') and metrics.names:
                class_name = metrics.names[i] if isinstance(metrics.names, list) else metrics.names.get(i, f"class_{i}")
            
            results[f"{class_name}/bbox_precision"] = float(metrics.box.p[i])
            results[f"{class_name}/bbox_recall"] = float(metrics.box.r[i])
            results[f"{class_name}/bbox_mAP50"] = float(metrics.box.ap50[i])
            results[f"{class_name}/bbox_mAP50-95"] = float(metrics.box.ap[i])

        # Метрики для масок (если доступны)
        if hasattr(metrics, "seg") and metrics.seg is not None:
            results["mask_mAP50"] = float(metrics.seg.map50)
            results["mask_mAP50-95"] = float(metrics.seg.map)
            results["mask_precision"] = float(metrics.seg.p.mean())
            results["mask_recall"] = float(metrics.seg.r.mean())

            # Вычисляем Dice coefficient для масок
            mask_p = results["mask_precision"]
            mask_r = results["mask_recall"]
            if mask_p + mask_r > 0:
                results["mask_dice"] = 2 * mask_p * mask_r / (mask_p + mask_r)
            else:
                results["mask_dice"] = 0.0

            # Метрики масок по каждому классу
            for i in range(num_classes):
                class_name = f"class_{i}"
                if hasattr(metrics, 'names') and metrics.names:
                    class_name = metrics.names[i] if isinstance(metrics.names, list) else metrics.names.get(i, f"class_{i}")
                
                results[f"{class_name}/mask_precision"] = float(metrics.seg.p[i])
                results[f"{class_name}/mask_recall"] = float(metrics.seg.r[i])
                results[f"{class_name}/mask_mAP50"] = float(metrics.seg.ap50[i])
                results[f"{class_name}/mask_mAP50-95"] = float(metrics.seg.ap[i])
                
                # Dice coefficient для класса
                mp = results[f"{class_name}/mask_precision"]
                mr = results[f"{class_name}/mask_recall"]
                if mp + mr > 0:
                    results[f"{class_name}/mask_dice"] = 2 * mp * mr / (mp + mr)
                else:
                    results[f"{class_name}/mask_dice"] = 0.0
        else:
            results["mask_mAP50"] = 0.0
            results["mask_mAP50-95"] = 0.0
            results["mask_precision"] = 0.0
            results["mask_recall"] = 0.0
            results["mask_dice"] = 0.0

    return results


def run_inference_and_visualize(
    model: Any,
    images_dir: Path,
    labels_dir: Path,
    save_dir: Path,
    max_images: int = 50,
    conf: float = 0.25,
    task: str = "detect",
    class_names: Optional[List[str]] = None,
    visualize: bool = False,
    make_visualize_comparison: bool = False,
) -> None:
    """
    Выполнить инференс модели на изображениях и сохранить визуализации.

    Инференс выполняется один раз для каждого изображения, результаты
    используются для всех запрошенных типов визуализации.

    Args:
        model: YOLO модель
        images_dir: Директория с изображениями
        labels_dir: Директория с метками
        save_dir: Директория для сохранения визуализаций
        max_images: Максимальное количество изображений
        conf: Порог уверенности
        task: Тип задачи ('detect' или 'segment')
        class_names: Названия классов
        visualize: Сохранять предсказания (bbox/mask на изображении)
        make_visualize_comparison: Сохранять сравнение Prediction vs GT
    """

    # Получаем список изображений
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if len(image_files) == 0:
        print(f" Не найдено изображений в {images_dir}")
        return

    # Выбираем случайные изображения
    n_images = min(max_images, len(image_files))
    selected_images = random.sample(image_files, n_images)

    print(f"\nВизуализация {n_images} изображений...")

    # Создаём директории для сохранения
    vis_dir = save_dir / "visualizations" if visualize else None
    compare_dir = save_dir / "visualizations_compare" if make_visualize_comparison else None

    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if compare_dir:
        compare_dir.mkdir(parents=True, exist_ok=True)

    # Выполняем инференс и визуализацию
    for i, img_file in enumerate(selected_images, 1):
        img_path = images_dir / img_file
        label_path = labels_dir / img_path.with_suffix(".txt").name

        if not label_path.exists():
            continue

        print(f"  [{i}/{n_images}] {img_file}...", end=" ")

        # Загружаем изображение
        image = cv2.imread(str(img_path))
        if image is None:
            print(" Не удалось загрузить изображение")
            continue

        # Выполняем инференс (ОДИН РАЗ для всех визуализаций)
        results = model.predict(str(img_path), conf=conf, verbose=False)
        result = results[0]

        # Сохраняем визуализации
        if visualize and vis_dir:
            output_path = vis_dir / f"pred_{img_file.replace('.jpg', '.png').replace('.jpeg', '.png')}"
            visualize_predictions(
                image=image,
                result=result,
                save_path=str(output_path),
                task=task,
                class_names=class_names,
            )

        if make_visualize_comparison and compare_dir and label_path.exists():
            output_path = compare_dir / f"compare_{img_file.replace('.jpg', '.png').replace('.jpeg', '.png')}"
            visualize_comparison(
                image=image,
                result=result,
                ground_truth_path=str(label_path),
                save_path=str(output_path),
                class_names=class_names,
                task=task,
            )

        print("Done")

    print(f"\n  Визуализации сохранены в {save_dir}")


def save_results(
    results: Dict[str, Any],
    save_dir: str,
    config: Dict[str, Any],
) -> None:
    """
    Сохранить результаты бенчмаркинга.

    Args:
        results: Словарь с результатами
        save_dir: Директория для сохранения
        config: Конфигурация
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Группируем метрики по классам
    common_metrics = {k: v for k, v in results.items() if "/" not in k}
    class_metrics = {k: v for k, v in results.items() if "/" in k}
    
    # Создаём структуру по классам
    metrics_by_class = {}
    for key, value in class_metrics.items():
        class_name, metric_name = key.split("/", 1)
        if class_name not in metrics_by_class:
            metrics_by_class[class_name] = {}
        metrics_by_class[class_name][metric_name] = value

    # Сохраняем метрики в JSON
    metrics_output = {
        "common_metrics": common_metrics,
        "class_metrics": metrics_by_class,
    }

    metrics_path = save_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2, ensure_ascii=False)

    # Сохраняем конфигурацию
    config_path = save_path / "benchmark_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\nРезультаты сохранены в {save_dir}")
    print(f"  - Метрики: {metrics_path}")
    print(f"  - Конфигурация: {config_path}")


def benchmark(config_path: str) -> None:
    """
    Запустить бенчмаркинг модели.

    Args:
        config_path: Путь к конфигурационному файлу
    """
    print("=" * 60)
    print("YOLO Template - Бенчмаркинг модели")
    print("=" * 60)

    # Загружаем конфигурацию
    print(f"\n[1/5] Загрузка конфигурации из {config_path}...")
    config = load_config(config_path)

    task = config.get("task", "detect")
    model_path = config.get("model", None)
    data_path = config.get("data", None)

    # Разрешаем относительные пути относительно корня проекта
    project_root = Path(__file__).parent.parent
    if data_path and not Path(data_path).is_absolute():
        data_path = str(project_root / data_path)

    print(f"  Задача: {task}")
    print(f"  Модель: {model_path}")
    print(f"  Датасет: {data_path}")

    # Проверяем наличие модели и датасета
    if model_path and not Path(model_path).exists():
        print(f"\n Ошибка: Модель не найдена по пути {model_path}")
        sys.exit(1)

    if data_path and not Path(data_path).exists():
        print(f"\n Ошибка: Датасет не найден по пути {data_path}")
        sys.exit(1)

    # Загружаем модель
    print("\n[2/5] Загрузка модели...")
    from ultralytics import YOLO

    model = YOLO(model_path)
    print(f"  Модель загружена: {model_path}")

    # Определяем директорию для сохранения результатов (в папке с весами модели)
    model_dir = Path(model_path).parent.parent
    save_dir = model_dir / "benchmark"

    # Запускаем бенчмаркинг
    print("\n[3/5] Запуск бенчмаркинга...")

    metrics = run_benchmark(
        model=model,
        data_path=data_path,
        task=task,
        imgsz=config.get("imgsz", 640),
        batch=config.get("batch", 16),
        conf=config.get("conf", 0.25),
        iou=config.get("iou", 0.45),
        split=config.get("split", "test"),
        device=config.get("device", None),
        save_dir=save_dir,
        plots=config.get("plots", True),
        save_json=config.get("save_json", False),
        amp=config.get("amp", True),
    )

    # Выводим метрики
    print("\n" + "=" * 100)
    print("Результаты бенчмаркинга")
    print("=" * 100)
    
    # Сначала общие метрики
    print("\n ОБЩИЕ МЕТРИКИ:")
    print("-" * 100)
    common_metrics = {k: v for k, v in metrics.items() if "/" not in k}
    for name, value in common_metrics.items():
        print(f"  {name:30s}: {value:.4f}")
    
    # Метрики по классам - группируем по классам
    class_metrics = {k: v for k, v in metrics.items() if "/" in k}
    
    # Извлекаем уникальные имена классов
    class_names_list = sorted(set(k.split("/")[0] for k in class_metrics.keys()))
    
    if class_names_list:
        print(f"\n МЕТРИКИ ПО КЛАССАМ:")
        print("-" * 100)
        
        # Для детекции
        if task == "detect":
            # Заголовок таблицы
            print(f"\n{'Класс':<15} {'Precision':>12} {'Recall':>12} {'mAP50':>12} {'mAP50-95':>12} {'F1':>12}")
            print("-" * 75)
            
            for cls_name in class_names_list:
                p = class_metrics.get(f"{cls_name}/precision", 0.0)
                r = class_metrics.get(f"{cls_name}/recall", 0.0)
                ap50 = class_metrics.get(f"{cls_name}/mAP50", 0.0)
                ap95 = class_metrics.get(f"{cls_name}/mAP50-95", 0.0)
                f1 = class_metrics.get(f"{cls_name}/f1", 0.0)
                print(f"{cls_name:<15} {p:>12.4f} {r:>12.4f} {ap50:>12.4f} {ap95:>12.4f} {f1:>12.4f}")
        
        # Для сегментации
        elif task == "segment":
            # Bbox метрики
            print(f"\n{'BBOX Метрики':^100}")
            print(f"{'Класс':<15} {'Precision':>12} {'Recall':>12} {'mAP50':>12} {'mAP50-95':>12}")
            print("-" * 100)
            
            for cls_name in class_names_list:
                p = class_metrics.get(f"{cls_name}/bbox_precision", 0.0)
                r = class_metrics.get(f"{cls_name}/bbox_recall", 0.0)
                ap50 = class_metrics.get(f"{cls_name}/bbox_mAP50", 0.0)
                ap95 = class_metrics.get(f"{cls_name}/bbox_mAP50-95", 0.0)
                print(f"{cls_name:<15} {p:>12.4f} {r:>12.4f} {ap50:>12.4f} {ap95:>12.4f}")
            
            # Mask метрики
            print(f"\n{'MASK Метрики':^100}")
            print(f"{'Класс':<15} {'Precision':>12} {'Recall':>12} {'mAP50':>12} {'mAP50-95':>12} {'Dice':>12}")
            print("-" * 100)
            
            for cls_name in class_names_list:
                p = class_metrics.get(f"{cls_name}/mask_precision", 0.0)
                r = class_metrics.get(f"{cls_name}/mask_recall", 0.0)
                ap50 = class_metrics.get(f"{cls_name}/mask_mAP50", 0.0)
                ap95 = class_metrics.get(f"{cls_name}/mask_mAP50-95", 0.0)
                dice = class_metrics.get(f"{cls_name}/mask_dice", 0.0)
                print(f"{cls_name:<15} {p:>12.4f} {r:>12.4f} {ap50:>12.4f} {ap95:>12.4f} {dice:>12.4f}")
    
    print("=" * 100)

    # Визуализация (инференс выполняется один раз для всех типов)
    if config.get("visualize", False) or config.get("make_visualize_comparison", False):
        print("\n[4/5] Визуализация...")

        # Получаем путь к изображениям и меткам из data.yaml
        images_dir = None
        labels_dir = None
        class_names = None
        split = config.get("split", "test")

        if data_path:
            data_yaml_path = Path(data_path)
            if data_yaml_path.is_file():
                with open(data_yaml_path, "r") as f:
                    data_config = yaml.safe_load(f)
                data_root = data_yaml_path.parent
            elif (data_yaml_path / "data.yaml").exists():
                with open(data_yaml_path / "data.yaml", "r") as f:
                    data_config = yaml.safe_load(f)
                data_root = data_yaml_path
            else:
                data_config = {}
                data_root = None

            class_names = data_config.get("names", None)

            # Получаем путь к изображениям и меткам для сплита
            if data_root:
                split_path = data_config.get(split, f"images/{split}")
                if split_path.startswith("images/"):
                    images_dir = data_root / split_path
                    labels_dir = data_root / "labels" / split_path.replace("images/", "")
                else:
                    images_dir = data_root / split_path
                    labels_dir = data_root / "labels" / split_path

        if images_dir and labels_dir and images_dir.exists():
            run_inference_and_visualize(
                model=model,
                images_dir=images_dir,
                labels_dir=labels_dir,
                save_dir=save_dir,
                max_images=config.get("max_visualizations", 50),
                conf=config.get("conf", 0.25),
                task=task,
                class_names=class_names,
                visualize=config.get("visualize", False),
                make_visualize_comparison=config.get("make_visualize_comparison", False),
            )
        else:
            print(f" Не найдены директории с изображениями/метками для сплита '{split}'")

    # Сохраняем результаты
    print("\n[5/5] Сохранение результатов...")

    save_results(
        results=metrics,
        save_dir=save_dir,
        config=config,
    )

    print("\n Бенчмаркинг завершен успешно!")


def main() -> None:
    """Точка входа для скрипта бенчмаркинга."""
    parser = argparse.ArgumentParser(
        description="Бенчмаркинг YOLO модели",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
    python scripts/benchmark.py --config configs/benchmark/benchmark.yaml
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Путь к YAML конфигурационному файлу",
    )

    args = parser.parse_args()

    # Проверяем наличие конфигурационного файла
    if not Path(args.config).exists():
        print(f" Ошибка: Конфигурационный файл не найден: {args.config}")
        sys.exit(1)

    benchmark(args.config)


if __name__ == "__main__":
    main()
