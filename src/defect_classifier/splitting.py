"""Train/test splitting strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split


@dataclass(slots=True)
class SplitResult:
    development: pd.DataFrame
    test: pd.DataFrame
    duplicate_overlap_rows_removed: int = 0


class PurgedTimeSeriesSplit:
    """Expanding-window CV that purges duplicate groups from validation folds."""

    def __init__(self, n_splits: int, group_column: str = "duplicate_group"):
        self.n_splits = n_splits
        self.group_column = group_column

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        base = TimeSeriesSplit(n_splits=self.n_splits)
        for train_indices, validation_indices in base.split(X, y):
            if not hasattr(X, "columns") or self.group_column not in X.columns:
                yield train_indices, validation_indices
                continue
            train_groups = set(X.iloc[train_indices][self.group_column].astype(str))
            validation_groups = X.iloc[validation_indices][self.group_column].astype(str)
            keep = ~validation_groups.isin(train_groups)
            purged_validation = validation_indices[keep.to_numpy()]
            if not len(purged_validation):
                raise ValueError("Duplicate purge removed an entire temporal validation fold")
            yield train_indices, purged_validation


def cv_fold_distribution(
    frame: pd.DataFrame,
    target_column: str,
    n_splits: int,
    group_column: str = "duplicate_group",
) -> pd.DataFrame:
    """Return auditable class counts for the exact CV folds used by model search."""

    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, group_column=group_column)
    rows = []
    features = frame.drop(columns=[target_column])
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(features, frame[target_column]), start=1
    ):
        for partition, indices in (
            ("train", train_indices),
            ("validation", validation_indices),
        ):
            counts = frame.iloc[indices][target_column].value_counts()
            for label, count in counts.items():
                rows.append(
                    {
                        "fold": fold,
                        "partition": partition,
                        "severity": label,
                        "count": int(count),
                        "rows_in_partition": int(len(indices)),
                    }
                )
    return pd.DataFrame(rows)


def remove_exact_duplicates(frame: pd.DataFrame, subset: list[str]) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicates on the selected subset."""

    before = len(frame)
    deduplicated = frame.drop_duplicates(subset=subset, keep="first").copy()
    return deduplicated, before - len(deduplicated)


def chronological_holdout(
    frame: pd.DataFrame,
    date_column: str,
    target_column: str,
    train_fraction: float,
    allow_missing_dates_fallback: bool = False,
) -> SplitResult:
    """Split by ascending date order into earliest development and latest test sets."""

    if date_column not in frame.columns:
        if not allow_missing_dates_fallback:
            raise ValueError(f"Chronological split requires a '{date_column}' column")
        return stratified_random_holdout(
            frame, target_column=target_column, train_fraction=train_fraction, random_state=42
        )

    ordered = frame.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column], errors="coerce", utc=True)
    if ordered[date_column].isna().all():
        if not allow_missing_dates_fallback:
            raise ValueError("Chronological split requires at least one parseable timestamp")
        return stratified_random_holdout(
            frame, target_column=target_column, train_fraction=train_fraction, random_state=42
        )
    if ordered[date_column].isna().any():
        missing_count = int(ordered[date_column].isna().sum())
        if not allow_missing_dates_fallback:
            raise ValueError(
                "Chronological split found "
                f"{missing_count} rows with missing/unparseable timestamps"
            )
        return stratified_random_holdout(
            frame, target_column=target_column, train_fraction=train_fraction, random_state=42
        )

    ordered = ordered.sort_values(date_column, kind="mergesort").reset_index(drop=True)
    split_index = int(np.floor(len(ordered) * train_fraction))
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Chronological split produced an empty development or test set")
    development = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()
    _validate_class_presence(development, test, target_column)
    return SplitResult(development=development, test=test)


def stratified_random_holdout(
    frame: pd.DataFrame,
    target_column: str,
    train_fraction: float,
    random_state: int,
) -> SplitResult:
    """Split with a reproducible stratified random holdout."""

    train_frame, test_frame = train_test_split(
        frame,
        train_size=train_fraction,
        random_state=random_state,
        stratify=frame[target_column],
    )
    return SplitResult(
        development=train_frame.reset_index(drop=True), test=test_frame.reset_index(drop=True)
    )


def _validate_class_presence(
    development: pd.DataFrame, test: pd.DataFrame, target_column: str
) -> None:
    dev_labels = set(development[target_column].astype(str).str.strip().str.lower())
    test_labels = set(test[target_column].astype(str).str.strip().str.lower())
    full_labels = dev_labels | test_labels
    missing_in_dev = sorted(full_labels - dev_labels)
    if missing_in_dev:
        raise ValueError(
            "Chronological split contains classes in the future test period that cannot be "
            f"learned from development data: missing_in_dev={missing_in_dev}"
        )


def purge_duplicate_group_overlap(split: SplitResult, group_column: str) -> tuple[SplitResult, int]:
    """Remove test rows whose duplicate group is already represented in development."""

    if group_column not in split.development.columns or group_column not in split.test.columns:
        return split, 0
    development_groups = set(split.development[group_column].dropna().astype(str))
    overlap = split.test[group_column].astype(str).isin(development_groups)
    removed = int(overlap.sum())
    test = split.test.loc[~overlap].reset_index(drop=True).copy()
    if test.empty:
        raise ValueError("Duplicate-group leakage prevention removed the entire test set")
    return (
        SplitResult(
            development=split.development,
            test=test,
            duplicate_overlap_rows_removed=removed,
        ),
        removed,
    )
