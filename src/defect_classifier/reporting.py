"""Persistence helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_directory, write_csv, write_json


def make_experiment_dir(root_dir: str | Path, experiment_id: str) -> Path:
    """Create a unique experiment directory tree."""

    experiment_root = ensure_directory(Path(root_dir) / "experiments" / experiment_id)
    for subdir in ["artifacts", "metrics", "predictions", "figures", "tables", "logs"]:
        ensure_directory(experiment_root / subdir)
    return experiment_root


def save_joblib(path: str | Path, obj: Any) -> None:
    """Persist a Python object with joblib."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    joblib.dump(obj, output_path)


def save_confusion_matrix_figure(path: str | Path, matrix: pd.DataFrame, title: str) -> None:
    """Render a confusion matrix figure using matplotlib only."""

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{matrix.iat[row_index, col_index]:.2f}"
                if isinstance(matrix.iat[row_index, col_index], float)
                else str(matrix.iat[row_index, col_index]),
                ha="center",
                va="center",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_frame(path: str | Path, frame: pd.DataFrame) -> None:
    write_csv(path, frame)


def save_payload(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
