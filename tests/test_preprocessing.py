from __future__ import annotations

import pandas as pd

from defect_classifier.models import build_pipeline
from defect_classifier.preprocessing import clean_text


def test_text_cleaning_replaces_urls_and_emails():
    text = clean_text(
        "Visit https://example.com and email bug@example.com",
        {
            "lowercase": True,
            "remove_html": True,
            "replace_urls": True,
            "replace_emails": True,
            "remove_code_blocks": False,
            "remove_stack_traces": False,
            "normalize_unicode": True,
            "normalize_whitespace": True,
        },
    )
    assert "[URL]" in text
    assert "[EMAIL]" in text


def test_pipeline_preprocessing_is_inside_pipeline(sample_config_path):
    from defect_classifier.config import load_config

    config = load_config(sample_config_path)
    pipeline = build_pipeline("LogisticRegression", config)
    assert "text" in pipeline.named_steps
    assert "tfidf" in pipeline.named_steps
    assert not hasattr(pipeline.named_steps["tfidf"], "vocabulary_")

    frame = pd.DataFrame(
        {
            "summary": ["Crash on startup", "Minor UI issue"],
            "description": ["Application crashes immediately.", "Tooltip overlaps."],
        }
    )
    target = ["blocker", "minor"]
    pipeline.fit(frame, target)
    assert hasattr(pipeline.named_steps["tfidf"], "vocabulary_")
