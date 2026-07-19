from __future__ import annotations

import pandas as pd

from defect_classifier.models import build_pipeline


def test_model_pipeline_fits(sample_config_path):
    from defect_classifier.config import load_config

    config = load_config(sample_config_path)
    pipeline = build_pipeline("LogisticRegression", config)
    frame = pd.DataFrame(
        {
            "summary": ["Crash on startup", "Minor UI issue"],
            "description": ["Application crashes immediately.", "Tooltip overlaps."],
        }
    )
    target = ["blocker", "minor"]
    pipeline.fit(frame, target)
    predictions = pipeline.predict(frame)
    assert len(predictions) == 2
