from __future__ import annotations

import pandas as pd
import pytest

from defect_classifier.splitting import (
    PurgedTimeSeriesSplit,
    SplitResult,
    chronological_holdout,
    purge_duplicate_group_overlap,
    stratified_random_holdout,
)


def test_chronological_split(sample_frame):
    frame = sample_frame.rename(columns=str.lower)
    result = chronological_holdout(frame, "creation time", "severity", 0.8)
    assert len(result.development) == 16
    assert len(result.test) == 4
    assert result.development["creation time"].max() < result.test["creation time"].min()


def test_stratified_split(sample_frame):
    frame = sample_frame.rename(columns=str.lower)
    result = stratified_random_holdout(frame, "severity", 0.8, 42)
    assert len(result.development) == 16
    assert len(result.test) == 4


def test_chronological_split_rejects_partial_missing_dates(sample_frame):
    frame = sample_frame.rename(columns=str.lower)
    frame.loc[0, "creation time"] = pd.NaT
    with pytest.raises(ValueError, match="missing/unparseable"):
        chronological_holdout(frame, "creation time", "severity", 0.8)


def test_duplicate_groups_are_not_shared_between_partitions():
    split = SplitResult(
        development=pd.DataFrame({"duplicate_group": ["1", "2"]}),
        test=pd.DataFrame({"duplicate_group": ["2", "3"]}),
    )
    purged, removed = purge_duplicate_group_overlap(split, "duplicate_group")
    assert removed == 1
    assert purged.test["duplicate_group"].tolist() == ["3"]


def test_temporal_cv_purges_duplicate_groups_from_validation():
    frame = pd.DataFrame(
        {
            "feature": range(12),
            "duplicate_group": ["a", "b", "c", "a", "d", "e", "b", "f", "g", "h", "i", "j"],
        }
    )
    splitter = PurgedTimeSeriesSplit(n_splits=3)
    for train, validation in splitter.split(frame):
        train_groups = set(frame.iloc[train]["duplicate_group"])
        validation_groups = set(frame.iloc[validation]["duplicate_group"])
        assert train_groups.isdisjoint(validation_groups)
