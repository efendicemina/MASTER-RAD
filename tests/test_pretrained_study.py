from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from defect_classifier.development_study import redact
from defect_classifier.pretrained_study import (
    EXPECTED_FOLD_HASH,
    LABELS,
    MacroF1EarlyStopping,
    _fixed_metrics,
    class_weights_from_training,
    embedding_cache_fingerprint,
    focal_loss,
    freeze_encoder,
    label_mapping,
    load_pretrained_development,
    tokenize_preserving_summary,
    train_torch_classifier,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [index + 10 for index, _ in enumerate(str(text).split())]

    def num_special_tokens_to_add(self, pair=True):
        assert pair
        return 3

    def prepare_for_model(self, first, pair_ids, **kwargs):
        del kwargs
        values = [101, *first, 102, *pair_ids, 102]
        values += [0] * (256 - len(values))
        return {"input_ids": values, "attention_mask": [int(value != 0) for value in values]}


def test_all_held_out_artifact_names_are_rejected(tmp_path: Path):
    for name in [
        "test_split.csv",
        "test_predictions.csv",
        "test_metrics.json",
        "held_out_labels.csv",
    ]:
        path = tmp_path / name
        path.write_text("forbidden", encoding="utf-8")
        with pytest.raises(ValueError, match="prohibited|Only"):
            load_pretrained_development(path)


def test_frozen_fold_identity_is_protocol_constant():
    assert EXPECTED_FOLD_HASH == "5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903"


def test_label_mapping_is_deterministic():
    assert label_mapping() == {label: index for index, label in enumerate(LABELS)}


def test_summary_is_preserved_before_description_truncation():
    tokenizer = TinyTokenizer()
    summary = " ".join(f"summary{index}" for index in range(20))
    description = " ".join(f"description{index}" for index in range(400))
    result = tokenize_preserving_summary(tokenizer, summary, description)
    assert result["summary_tokens_kept"] == 20
    assert result["description_tokens_kept"] == 233
    assert len(result["input_ids"]) == 256


def test_embedding_fingerprint_is_deterministic_and_revision_sensitive(monkeypatch):
    first = embedding_cache_fingerprint("dataset", "fold", "preprocessing")
    second = embedding_cache_fingerprint("dataset", "fold", "preprocessing")
    assert first == second
    monkeypatch.setattr("defect_classifier.pretrained_study.MODEL_REVISION", "different")
    assert embedding_cache_fingerprint("dataset", "fold", "preprocessing") != first


def test_encoder_is_frozen():
    encoder = freeze_encoder(torch.nn.Linear(3, 2))
    assert encoder.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_dense_embeddings_work_with_both_frozen_classifiers():
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    features = np.eye(12, dtype=np.float32)
    labels = LABELS * 2
    for classifier in [
        LogisticRegression(class_weight="balanced", max_iter=1000),
        LinearSVC(class_weight="balanced"),
    ]:
        assert len(classifier.fit(features, labels).predict(features)) == 12


def test_class_weights_use_training_labels_only():
    train = LABELS + ["normal"] * 6
    weights = class_weights_from_training(train)
    assert weights[LABELS.index("normal")] < weights[LABELS.index("blocker")]
    with pytest.raises(ValueError):
        class_weights_from_training(["normal"] * 10)


def test_focal_loss_is_finite():
    logits = torch.tensor([[2.0, 0, 0, 0, 0, 0], [0, 2.0, 0, 0, 0, 0]])
    targets = torch.tensor([0, 1])
    assert torch.isfinite(focal_loss(logits, targets, torch.ones(6)))


def test_early_stopping_selects_highest_macro_f1_epoch():
    stopper = MacroF1EarlyStopping(patience=1)
    assert stopper.update(0.2, 1) is False
    assert stopper.update(0.3, 2) is False
    assert stopper.update(0.25, 3) is False
    assert stopper.update(0.24, 4) is True
    assert stopper.best_epoch == 2


@pytest.mark.parametrize("loss_type", ["weighted_cross_entropy", "focal"])
def test_cpu_smoke_training_and_macro_f1_checkpoint(loss_type):
    torch.manual_seed(42)
    features = torch.randn(24, 5)
    targets = torch.tensor(list(range(6)) * 4)
    loader = DataLoader(TensorDataset(features, targets), batch_size=6, shuffle=False)
    model = torch.nn.Linear(5, 6)
    result = train_torch_classifier(
        model,
        loader,
        loader,
        torch.ones(6),
        loss_type,
        epochs=2,
        learning_rate=1e-2,
        gradient_accumulation=1,
    )
    assert result["best_epoch"] in {1, 2}
    assert 0 <= result["best_macro_f1"] <= 1


def test_persisted_example_redaction():
    text = redact("contact a@example.com at http://example.com/report")
    assert "example.com" not in text
    assert "[EMAIL]" in text and "[URL]" in text


def test_fixed_label_metrics_keep_absent_classes():
    metrics = _fixed_metrics(np.asarray(["normal"]), np.asarray(["normal"]))
    assert metrics["macro_f1"] == pytest.approx(1 / 6)
    assert metrics["f1_blocker"] == 0


def test_no_pretrained_challenger_is_created_without_material_improvement():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "configs" / "eclipse_training_mylyn_pretrained_challenger.yaml").exists()
    assert (root / "docs" / "mylyn_pretrained_no_improvement.md").is_file()
