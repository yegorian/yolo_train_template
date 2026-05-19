"""
Утилиты для построения графиков.

Функции для визуализации метрик обучения и результатов бенчмаркинга.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_training_metrics(
    results_csv_path: str,
    save_path: str,
    figsize: Tuple[int, int] = (15, 15),
    dpi: int = 150,
) -> None:
    """
    Построить графики метрик обучения из results.csv.

    Args:
        results_csv_path: Путь к файлу results.csv от Ultralytics
        save_path: Путь для сохранения графика
        figsize: Размер фигуры (ширина, высота)
        dpi: DPI для сохранения
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib и seaborn требуются для построения графиков")

    import pandas as pd

    # Чтение результатов
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()

    # Определение доступных колонок
    loss_cols = [
        col
        for col in df.columns
        if "loss" in col.lower() and "val" not in col.lower()
    ]
    val_loss_cols = [col for col in df.columns if "val" in col.lower() and "loss" in col.lower()]
    metric_cols = [col for col in df.columns if "metrics" in col.lower() or "map" in col.lower()]

    # Определение количества графиков
    n_plots = len(loss_cols) + len(val_loss_cols) + len(metric_cols)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten() if n_plots > 1 else [axs]

    plot_idx = 0

    # Графики потерь
    for col in loss_cols:
        if col in df.columns:
            axs[plot_idx].plot(df["epoch"], df[col], label=col)
            axs[plot_idx].set_title(col.replace("train/", ""))
            axs[plot_idx].set_xlabel("Epoch")
            axs[plot_idx].set_ylabel("Loss")
            axs[plot_idx].legend()
            axs[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

    # Графики валидационных потерь
    for col in val_loss_cols:
        if col in df.columns:
            axs[plot_idx].plot(df["epoch"], df[col], label=col, color="orange")
            axs[plot_idx].set_title(col.replace("val/", ""))
            axs[plot_idx].set_xlabel("Epoch")
            axs[plot_idx].set_ylabel("Loss")
            axs[plot_idx].legend()
            axs[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

    # Графики метрик
    for col in metric_cols:
        if col in df.columns:
            axs[plot_idx].plot(df["epoch"], df[col], label=col)
            axs[plot_idx].set_title(col.replace("metrics/", ""))
            axs[plot_idx].set_xlabel("Epoch")
            axs[plot_idx].set_ylabel("Metric")
            axs[plot_idx].legend()
            axs[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

    # Удаляем пустые подграфики
    for i in range(plot_idx, len(axs)):
        fig.delaxes(axs[i])

    plt.suptitle("Training Metrics", fontsize=16, y=1.02)
    plt.tight_layout()

    # Сохранение
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Loss Curves",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
) -> None:
    """
    Построить графики кривых потерь.

    Args:
        train_losses: Список тренировочных потерь по эпохам
        val_losses: Список валидационных потерь (опционально)
        save_path: Путь для сохранения (если None, показать график)
        title: Заголовок графика
        figsize: Размер фигуры
        dpi: DPI для сохранения
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib и seaborn требуются для построения графиков")

    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3)

    if val_losses is not None:
        ax.plot(epochs, val_losses, label="Val Loss", marker="s", markersize=3)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 150,
    cmap: str = "Blues",
) -> None:
    """
    Построить матрицу ошибок (confusion matrix).

    Args:
        confusion_matrix: Матрица ошибок [n_classes, n_classes]
        class_names: Названия классов
        save_path: Путь для сохранения (если None, показать график)
        title: Заголовок графика
        figsize: Размер фигуры
        dpi: DPI для сохранения
        cmap: Цветовая схема matplotlib
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib и seaborn требуются для построения графиков")

    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=figsize)

    # Нормализация для отображения процентов
    cm_normalized = confusion_matrix.astype("float") / (
        confusion_matrix.sum(axis=1, keepdims=True) + 1e-10
    )

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion"},
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Поворот подписей для лучшей читаемости
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Metrics Comparison",
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 150,
) -> None:
    """
    Построить сравнение метрик между разными моделями/конфигурациями.

    Args:
        metrics_dict: Словарь вида {model_name: {metric_name: value}}
        save_path: Путь для сохранения
        title: Заголовок графика
        figsize: Размер фигуры
        dpi: DPI для сохранения
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib и seaborn требуются для построения графиков")

    import pandas as pd

    # Преобразование в DataFrame
    df = pd.DataFrame(metrics_dict).T
    df = df.reset_index().rename(columns={"index": "Model"})

    # Melt для seaborn
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Value")

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric", ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 150,
) -> None:
    """
    Построить Precision-Recall кривую.

    Args:
        precision: Массив precision значений
        recall: Массив recall значений
        save_path: Путь для сохранения
        title: Заголовок графика
        figsize: Размер фигуры
        dpi: DPI для сохранения
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib и seaborn требуются для построения графиков")

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Добавляем значение AP
    ap = np.trapz(precision, recall)
    ax.text(
        0.95,
        0.05,
        f"AP = {ap:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
