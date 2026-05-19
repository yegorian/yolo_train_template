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
        --config configs/data/data_config.json

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