from __future__ import annotations

from pathlib import Path

from defect_classifier.within_project import (
    MODELS,
    PROJECT_ORDER,
    frozen_project_config,
    run_project,
)


def test_frozen_project_configuration_changes_only_dataset_identity():
    root = Path(__file__).resolve().parents[1]
    first = frozen_project_config("TPTP", root)
    second = frozen_project_config("Papyrus", root)
    for config in [first, second]:
        assert config["target_labels"] == [
            "blocker",
            "critical",
            "major",
            "normal",
            "minor",
            "trivial",
        ]
        assert config["features"]["representation"] == "summary_description"
        assert config["features"]["ngram_range"] == [1, 2]
        assert config["models"]["enabled"] == ["LogisticRegression"]
        assert config["models"]["class_weight"] == "balanced"
        assert config["split"]["strategy"] == "chronological"
        assert config["cv"]["strategy"] == "time_series"
    assert first["dataset"]["path"].endswith("TPTP.parquet")
    assert second["dataset"]["path"].endswith("Papyrus.parquet")


def test_project_and_model_order_is_frozen():
    assert PROJECT_ORDER == ["TPTP", "Papyrus", "PDE", "Equinox", "BIRT", "CDT", "JDT", "Platform"]
    assert MODELS == ["DummyClassifier", "MultinomialNB", "LogisticRegression", "LinearSVC"]


def test_completed_project_is_resumed_without_reexecution(tmp_path: Path):
    completed = tmp_path / "experiments" / "TPTP_existing" / "COMPLETED.json"
    completed.parent.mkdir(parents=True)
    completed.write_text("{}", encoding="utf-8")
    assert run_project("TPTP", tmp_path, tmp_path) == completed.parent
