from __future__ import annotations

from defect_classifier.config import load_config


def test_configuration_loading(sample_config_path):
    config = load_config(sample_config_path)
    assert config["split"]["train_fraction"] == 0.8
    assert config["features"]["representation"] == "summary_description"
