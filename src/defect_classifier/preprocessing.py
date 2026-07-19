"""Text normalization and sklearn-compatible transformers."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
STACK_TRACE_PATTERN = re.compile(
    r"(^\s*at\s+\S+|Exception in thread|Traceback \(most recent call last\):)", re.IGNORECASE
)
CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)


@dataclass(slots=True)
class CleaningOptions:
    lowercase: bool = True
    remove_html: bool = True
    replace_urls: bool = True
    replace_emails: bool = True
    remove_code_blocks: bool = False
    remove_stack_traces: bool = False
    normalize_unicode: bool = True
    normalize_whitespace: bool = True


def clean_text(value: object, options: CleaningOptions | dict[str, object]) -> str:
    """Conservatively normalize text for machine-learning use."""

    if isinstance(options, dict):
        options = CleaningOptions(**options)
    text = "" if value is None else str(value)
    if options.normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    if options.remove_code_blocks:
        text = CODE_BLOCK_PATTERN.sub(" ", text)
    if options.remove_stack_traces:
        text = STACK_TRACE_PATTERN.sub(" ", text)
    if options.remove_html:
        text = HTML_TAG_PATTERN.sub(" ", text)
    if options.lowercase:
        text = text.lower()
    if options.replace_urls:
        text = URL_PATTERN.sub(" [URL] ", text)
    if options.replace_emails:
        text = EMAIL_PATTERN.sub(" [EMAIL] ", text)
    if options.normalize_whitespace:
        text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


class TextCombiner(BaseEstimator, TransformerMixin):
    """Combine and clean summary and description text inside a pipeline."""

    def __init__(
        self,
        summary_column: str,
        description_column: str,
        representation: str,
        cleaning_options: dict[str, object] | CleaningOptions,
    ):
        self.summary_column = summary_column
        self.description_column = description_column
        self.representation = representation
        self.cleaning_options = cleaning_options

    def fit(self, X: pd.DataFrame, y: object = None) -> TextCombiner:
        return self

    def transform(self, X: pd.DataFrame) -> list[str]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TextCombiner expects a pandas DataFrame")
        if self.summary_column not in X.columns or self.description_column not in X.columns:
            raise ValueError("Input frame does not contain the required text columns")
        summary = X[self.summary_column].fillna("").astype(str)
        description = X[self.description_column].fillna("").astype(str)
        if self.representation == "summary":
            texts = summary
        elif self.representation == "description":
            texts = description
        else:
            texts = summary.str.cat(description, sep=" \n ")
        return [clean_text(value, self.cleaning_options) for value in texts]
