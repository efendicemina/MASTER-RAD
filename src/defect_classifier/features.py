"""Feature extraction configuration."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(config: dict[str, object]) -> TfidfVectorizer:
    """Construct a TF-IDF vectorizer from configuration values."""

    return TfidfVectorizer(
        analyzer=str(config["analyzer"]),
        ngram_range=tuple(config["ngram_range"]),  # type: ignore[arg-type]
        min_df=config["min_df"],
        max_df=config["max_df"],
        max_features=int(config["max_features"]),
        sublinear_tf=bool(config["sublinear_tf"]),
        lowercase=bool(config.get("lowercase", False)),
    )
