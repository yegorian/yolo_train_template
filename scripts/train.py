"""
Скрипт для обучения YOLO моделей.

Использование:
    python scripts/train.py --config configs/train/train_config.yaml

Аргументы:
    --config: Путь к YAML конфигурационному файлу
"""

import argparse
from typing import Any, Dict
import yaml
import sys
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings

# Отключение MLFLOW
settings.update({"mlflow": False})

# Добавляем корень проекта в sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Для допополнительных графиков
from yolo_template.visualization.plot_utils import plot_training_metrics

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


def prepare_training_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Подготовить аргументы для model.train() из конфигурации.

    Args:
        config: Словарь конфигурации

    Returns:
        Словарь аргументов для Ultralytics YOLO train()
    """
    # Параметры, которые передаются напрямую в model.train()
    # Согласно документации: https://docs.ultralytics.com/modes/train/
    train_args = {
        # Основные параметры
        "model": config.get("model"),
        "data": config.get("data", None),
        "epochs": config.get("epochs", 100),
        "time": config.get("time", None),
        "patience": config.get("patience", 100),
        "batch": config.get("batch", 16),
        "imgsz": config.get("imgsz", 640),
        "save": config.get("save", True),
        "save_period": config.get("save_period", -1),
        "cache": config.get("cache", False),
        "device": config.get("device", None),
        "workers": config.get("workers", 8),
        "project": config.get("project", None),
        "name": config.get("name", None),
        "exist_ok": config.get("exist_ok", False),
        "resume": config.get("resume", False),
        "pretrained": config.get("pretrained", True),
        "seed": config.get("seed", 0),
        "deterministic": config.get("deterministic", True),
        "verbose": config.get("verbose", True),
        "plots": config.get("plots", True),
        # Оптимизатор
        "optimizer": config.get("optimizer", "auto"),
        "lr0": config.get("lr0", 0.01),
        "lrf": config.get("lrf", 0.01),
        "momentum": config.get("momentum", 0.937),
        "weight_decay": config.get("weight_decay", 0.0005),
        "warmup_epochs": config.get("warmup_epochs", 3.0),
        "warmup_momentum": config.get("warmup_momentum", 0.8),
        "warmup_bias_lr": config.get("warmup_bias_lr", 0.1),
        "cos_lr": config.get("cos_lr", False),
        "close_mosaic": config.get("close_mosaic", 10),
        # Функции потерь
        "box": config.get("box", 7.5),
        "cls": config.get("cls", 0.5),
        "dfl": config.get("dfl", 1.5),
        "nbs": config.get("nbs", 64),
        # Аугментации
        "hsv_h": config.get("hsv_h", 0.015),
        "hsv_s": config.get("hsv_s", 0.7),
        "hsv_v": config.get("hsv_v", 0.4),
        "degrees": config.get("degrees", 0.0),
        "translate": config.get("translate", 0.1),
        "scale": config.get("scale", 0.5),
        "shear": config.get("shear", 0.0),
        "perspective": config.get("perspective", 0.0),
        "flipud": config.get("flipud", 0.0),
        "fliplr": config.get("fliplr", 0.5),
        "bgr": config.get("bgr", 0.0),
        "mosaic": config.get("mosaic", 1.0),
        "mixup": config.get("mixup", 0.0),
        "cutmix": config.get("cutmix", 0.0),
        "copy_paste": config.get("copy_paste", 0.0),
        "multi_scale": config.get("multi_scale", 0.0),
        "erasing": config.get("erasing", 0.4),
        # Валидация
        "val": config.get("val", True),
        "split": config.get("split", "val"),
        "amp": config.get("amp", True),
        # Дополнительные
        "single_cls": config.get("single_cls", False),
        "rect": config.get("rect", False),
        "freeze": config.get("freeze", None),
        "classes": config.get("classes", None),
        "fraction": config.get("fraction", 1.0),
        "profile": config.get("profile", False),
        "max_det": config.get("max_det", 300),
        "overlap_mask": config.get("overlap_mask", True),
        "mask_ratio": config.get("mask_ratio", 4),
        "dropout": config.get("dropout", 0.0),
    }

    # Удаляем None значения (но оставляем те, что нужны для Ultralytics)
    required_keys = {"model", "data", "device", "freeze", "classes", "auto_augment", "time", "project", "name"}
    train_args = {k: v for k, v in train_args.items() if v is not None or k in required_keys}

    return train_args

def train(config_path: str) -> None:
    """
    Запустить обучение модели.

    Args:
        config_path: Путь к конфигурационному файлу
    """
    print("=" * 60)
    print("YOLO Template - Обучение модели")
    print("=" * 60)

    # Загружаем конфигурацию
    print(f"\n[1/4] Загрузка конфигурации из {config_path}...")
    config = load_config(config_path)

    task = config.get("task", "detect")
    model_name = config.get("model", "yolov8n.pt")
    data_path = config.get("data", None)

    print(f"  Задача: {task}")
    print(f"  Модель: {model_name}")
    print(f"  Датасет: {data_path}")

    # Проверяем наличие датасета
    if data_path and not Path(data_path).exists():
        print(f"\n!!! Ошибка: Датасет не найден по пути {data_path}")
        print("   Убедитесь, что путь указан правильно в конфигурационном файле.")
        sys.exit(1)

    # Подготавливаем аргументы
    print("\n[2/4] Подготовка параметров обучения...")
    train_args = prepare_training_args(config)

    # Определяем директорию для сохранения результатов
    output_dir = train_args.get("project", "outputs/train")
    experiment_name = train_args.get("name", "experiment")
    full_output_dir = Path(output_dir) / experiment_name

    # Загружаем модель
    print("\n[3/4] Загрузка модели...")
    model = YOLO(model_name)
    print(f"  Модель загружена: {model_name}")

    # Запускаем обучение
    print("\n[4/4] Запуск обучения...")
    print("-" * 60)

    try:
        results = model.train(**train_args)

        print("-" * 60)
        print("\n Обучение завершено успешно!")

        # Выводим информацию о результатах
        print(f"\n Результаты сохранены в: {full_output_dir}")
        print(f"   - Веса модели: {full_output_dir / 'weights'}")
        print(f"   - Логи: {full_output_dir / 'results.csv'}")
        print(f"   - Графики: {full_output_dir / 'results.png'}")

        # Построение дополнительных графиков
        print("\nПостроение дополнительных графиков...")
        try:
            results_csv = full_output_dir / "results.csv"
            if results_csv.exists():
                plot_path = full_output_dir / "training_metrics.png"
                plot_training_metrics(
                    str(results_csv),
                    str(plot_path),
                )
                print(f"   Графики метрик сохранены в {plot_path}")
            else:
                print(f"   Файл с результатами обучения не найден")
        except Exception as e:
            print(f"   Не удалось построить графики: {e}")

    except Exception as e:
        print(f"\n Ошибка во время обучения: {e}")
        raise


def main() -> None:
    """Точка входа для скрипта обучения."""
    parser = argparse.ArgumentParser(
        description="Обучение YOLO модели",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
    python scripts/train.py --config configs/detection/train_config.yaml
    python scripts/train.py --config configs/segmentation/train_config.yaml
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

    train(args.config)


if __name__ == "__main__":
    main()
