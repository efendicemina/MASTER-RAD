"""MYLYN development-only pretrained representation study."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

from .development_study import (
    LABELS,
    folds_fingerprint,
    freeze_temporal_folds,
    load_development_only,
    redact,
)
from .reporting import save_confusion_matrix_figure
from .utils import package_versions, sha256_file, write_csv, write_json

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c21050a7ef692090620a6d037dd736908f9c7cf6"
MODEL_LICENSE = "Apache-2.0"
MAX_LENGTH = 256
SEED = 42
EXPECTED_FOLD_HASH = "5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903"
MYLYN_DEVELOPMENT_SHA256 = "8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5"
MATERIAL_THRESHOLD = 0.2463


def load_pretrained_development(path: str | Path) -> pd.DataFrame:
    """Load only the immutable MYLYN development artifact."""

    source = Path(path)
    forbidden = {"test_split.csv", "test_predictions.csv", "test_metrics.json"}
    if source.name in forbidden or "test" in source.name.lower():
        raise ValueError("Held-out test artifacts are prohibited")
    if source.name != "development_split.csv":
        raise ValueError("Only the MYLYN development split is accepted")
    if sha256_file(source).lower() != MYLYN_DEVELOPMENT_SHA256:
        raise ValueError("Development artifact hash does not match immutable MYLYN development")
    frame = load_development_only(source)
    if set(frame["severity"]) - set(LABELS):
        raise ValueError("Unexpected labels in MYLYN development")
    return frame


def label_mapping() -> dict[str, int]:
    return {label: index for index, label in enumerate(LABELS)}


def embedding_cache_fingerprint(
    dataset_hash: str, fold_hash: str, preprocessing_fingerprint: str
) -> str:
    payload = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dataset_hash": dataset_hash,
        "fold_hash": fold_hash,
        "preprocessing": preprocessing_fingerprint,
        "max_length": MAX_LENGTH,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def tokenize_preserving_summary(tokenizer, summary: str, description: str) -> dict[str, list[int]]:
    """Reserve token capacity for Summary before truncating Description."""

    summary_ids = tokenizer.encode(str(summary or ""), add_special_tokens=False)
    description_ids = tokenizer.encode(str(description or ""), add_special_tokens=False)
    special_count = tokenizer.num_special_tokens_to_add(pair=True)
    summary_budget = MAX_LENGTH - special_count
    kept_summary = summary_ids[:summary_budget]
    description_budget = max(0, MAX_LENGTH - special_count - len(kept_summary))
    kept_description = description_ids[:description_budget]
    prepared = tokenizer.prepare_for_model(
        kept_summary,
        pair_ids=kept_description,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=False,
        return_attention_mask=True,
    )
    prepared["summary_tokens_original"] = len(summary_ids)
    prepared["description_tokens_original"] = len(description_ids)
    prepared["summary_tokens_kept"] = len(kept_summary)
    prepared["description_tokens_kept"] = len(kept_description)
    return prepared


def class_weights_from_training(labels: list[str]) -> np.ndarray:
    counts = pd.Series(labels).value_counts()
    if set(counts.index) != set(LABELS):
        raise ValueError("Training fold must contain all fixed labels")
    return np.asarray([len(labels) / (len(LABELS) * counts[label]) for label in LABELS])


def focal_loss(logits, targets, weights, gamma: float = 2.0):
    import torch
    import torch.nn.functional as functional

    cross_entropy = functional.cross_entropy(logits, targets, weight=weights, reduction="none")
    probability = torch.exp(-cross_entropy)
    return ((1.0 - probability) ** gamma * cross_entropy).mean()


def train_torch_classifier(
    model,
    train_loader,
    validation_loader,
    weights,
    loss_type: str,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    gradient_accumulation: int = 4,
) -> dict[str, Any]:
    """Reusable weighted/focal training loop with macro-F1 checkpoint selection."""

    import torch
    import torch.nn.functional as functional

    if loss_type not in {"weighted_cross_entropy", "focal"}:
        raise ValueError("loss_type must be weighted_cross_entropy or focal")
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    stopper = MacroF1EarlyStopping(patience=1)
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            inputs, targets = batch
            logits = model(inputs)
            loss = (
                functional.cross_entropy(logits, targets, weight=weights)
                if loss_type == "weighted_cross_entropy"
                else focal_loss(logits, targets, weights)
            )
            (loss / gradient_accumulation).backward()
            if step % gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        predictions, targets_all = [], []
        with torch.inference_mode():
            for inputs, targets in validation_loader:
                predictions.extend(model(inputs).argmax(dim=1).cpu().tolist())
                targets_all.extend(targets.cpu().tolist())
        score = f1_score(
            targets_all,
            predictions,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        )
        history.append({"epoch": epoch, "validation_macro_f1": score})
        if score > stopper.best_score:
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if stopper.update(score, epoch):
            break
    if best_state is None:
        raise RuntimeError("No fine-tuning checkpoint was selected")
    model.load_state_dict(best_state)
    return {
        "best_epoch": stopper.best_epoch,
        "best_macro_f1": stopper.best_score,
        "history": history,
        "loss_type": loss_type,
    }


@dataclass(slots=True)
class MacroF1EarlyStopping:
    patience: int = 1
    best_score: float = float("-inf")
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, score: float, epoch: int) -> bool:
        if score > self.best_score:
            self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs > self.patience


def hardware_report(root: Path, output: Path) -> dict[str, Any]:
    import torch

    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(root)
    cuda = torch.cuda.is_available()
    report = {
        "operating_system": platform.platform(),
        "python_version": sys.version,
        "cpu": platform.processor(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": memory.total / 2**30,
        "available_ram_gb": memory.available / 2**30,
        "free_disk_gb": disk.free / 2**30,
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_vram_gb": torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else None,
        "full_fine_tuning_safe": False,
        "fine_tuning_block_reason": (
            "No CUDA GPU; full three-fold CPU fine-tuning risks excessive runtime and RAM pressure"
        ),
        "frozen_embedding_risk": "moderate CPU runtime; bounded batches and float32 cache",
        "packages": package_versions(["torch", "transformers", "accelerate", "tokenizers"]),
    }
    write_json(output / "hardware_report.json", report)
    markdown = ["# MYLYN pretrained hardware audit", ""]
    markdown.extend(f"- {key}: {value}" for key, value in report.items() if key != "packages")
    (output / "hardware_report.md").write_text("\n".join(markdown), encoding="utf-8")
    return report


def _load_encoder(cache_dir: Path):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir
    )
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir)
    freeze_encoder(model)
    return tokenizer, model


def freeze_encoder(model):
    """Put an encoder in inference-only mode."""

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _tokenize_frame(frame: pd.DataFrame, tokenizer) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    encoded, audit_rows = [], []
    for index, row in frame.iterrows():
        item = tokenize_preserving_summary(tokenizer, row["summary"], row["description"])
        encoded.append(item)
        total_original = item["summary_tokens_original"] + item["description_tokens_original"]
        total_kept = item["summary_tokens_kept"] + item["description_tokens_kept"]
        audit_rows.append(
            {
                "row_index": index,
                "summary_tokens": item["summary_tokens_original"],
                "description_tokens": item["description_tokens_original"],
                "total_content_tokens": total_original,
                "kept_content_tokens": total_kept,
                "truncated": total_kept < total_original,
                "empty_text": total_original == 0,
            }
        )
    arrays = {
        "input_ids": np.asarray([item["input_ids"] for item in encoded], dtype=np.int64),
        "attention_mask": np.asarray([item["attention_mask"] for item in encoded], dtype=np.int64),
    }
    if "token_type_ids" in encoded[0]:
        arrays["token_type_ids"] = np.asarray(
            [item["token_type_ids"] for item in encoded], dtype=np.int64
        )
    return arrays, pd.DataFrame(audit_rows)


def _encode(arrays: dict[str, np.ndarray], model, batch_size: int = 32) -> np.ndarray:
    import torch

    embeddings = []
    started = perf_counter()
    with torch.inference_mode():
        for start in range(0, len(arrays["input_ids"]), batch_size):
            inputs = {
                key: torch.from_numpy(value[start : start + batch_size])
                for key, value in arrays.items()
            }
            output = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
    result = np.vstack(embeddings)
    return result, perf_counter() - started


def _fixed_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    precision, recall, per_f1, support = precision_recall_fscore_support(
        true, predicted, labels=LABELS, zero_division=0
    )
    result = {
        "macro_f1": f1_score(true, predicted, labels=LABELS, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(true, predicted),
        "weighted_f1": f1_score(
            true, predicted, labels=LABELS, average="weighted", zero_division=0
        ),
    }
    for index, label in enumerate(LABELS):
        result[f"precision_{label}"] = precision[index]
        result[f"recall_{label}"] = recall[index]
        result[f"f1_{label}"] = per_f1[index]
        result[f"support_{label}"] = int(support[index])
    return result


def run_frozen_embeddings(development_path: Path, output: Path, cache_root: Path) -> None:
    frame = load_pretrained_development(development_path)
    folds = freeze_temporal_folds(frame)
    if folds_fingerprint(folds) != EXPECTED_FOLD_HASH:
        raise ValueError("Frozen fold identity mismatch")
    tokenizer, encoder = _load_encoder(cache_root / "huggingface")
    arrays, audit = _tokenize_frame(frame, tokenizer)
    write_csv(output / "tokenization_audit.csv", audit)
    write_json(
        output / "tokenization_summary.json",
        {
            "rows": len(audit),
            "max_length": MAX_LENGTH,
            "empty_text_count": int(audit.empty_text.sum()),
            "truncation_rate": float(audit.truncated.mean()),
            "token_percentiles": audit.total_content_tokens.quantile(
                [0.5, 0.9, 0.95, 0.99]
            ).to_dict(),
            "summary_preserved_rows": int(
                (audit.summary_tokens == audit.summary_tokens.clip(upper=MAX_LENGTH - 3)).sum()
            ),
        },
    )
    fingerprint = embedding_cache_fingerprint(
        MYLYN_DEVELOPMENT_SHA256, EXPECTED_FOLD_HASH, "summary_pair_description_v1"
    )
    cache = cache_root / "embeddings" / f"{fingerprint}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        embeddings = np.load(cache)["embeddings"]
        encode_seconds = 0.0
        cache_status = "reused"
    else:
        embeddings, encode_seconds = _encode(arrays, encoder)
        np.savez_compressed(cache, embeddings=embeddings)
        cache_status = "created"
    del encoder
    fold_rows, per_class, prediction_frames, runtime_rows = [], [], [], []
    classifiers = {
        "LogisticRegression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED
        ),
        "LinearSVC": LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
    }
    for fold in folds:
        train_index, validation_index = list(fold.train), list(fold.validation)
        for name, classifier in classifiers.items():
            started = perf_counter()
            classifier.fit(embeddings[train_index], frame.iloc[train_index].severity)
            fit_seconds = perf_counter() - started
            started = perf_counter()
            predicted = classifier.predict(embeddings[validation_index])
            inference_seconds = perf_counter() - started
            true = frame.iloc[validation_index].severity.to_numpy()
            metrics = _fixed_metrics(true, predicted)
            fold_rows.append({"classifier": name, "fold": fold.fold, **metrics})
            runtime_rows.append(
                {
                    "classifier": name,
                    "fold": fold.fold,
                    "fit_seconds": fit_seconds,
                    "inference_seconds": inference_seconds,
                    "encoder_seconds_shared": encode_seconds,
                    "peak_cpu_rss_bytes_observed": psutil.Process().memory_info().rss,
                    "embedding_cache_bytes": cache.stat().st_size,
                    "embedding_dimension": embeddings.shape[1],
                    "cache_status": cache_status,
                }
            )
            for label in LABELS:
                per_class.append(
                    {
                        "classifier": name,
                        "fold": fold.fold,
                        "severity": label,
                        "precision": metrics[f"precision_{label}"],
                        "recall": metrics[f"recall_{label}"],
                        "f1": metrics[f"f1_{label}"],
                        "support": metrics[f"support_{label}"],
                    }
                )
            predictions = frame.iloc[validation_index][
                ["id", "creation_time", "severity", "summary", "description"]
            ].copy()
            predictions["predicted"] = predicted
            predictions["classifier"] = name
            predictions["fold"] = fold.fold
            predictions["summary"] = predictions.summary.map(redact)
            predictions["description"] = predictions.description.map(
                lambda value: redact(value, 1000)
            )
            prediction_frames.append(predictions)
    results = pd.DataFrame(fold_rows)
    write_csv(output / "frozen_embedding_results.csv", results)
    write_csv(output / "frozen_embedding_per_class.csv", pd.DataFrame(per_class))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    write_csv(output / "frozen_embedding_oof_predictions.csv", predictions)
    write_csv(output / "frozen_embedding_runtime.csv", pd.DataFrame(runtime_rows))
    aggregate_rows = []
    for classifier, group in results.groupby("classifier"):
        row = {
            "approach": f"frozen_{classifier}",
            "mean_macro_f1": group.macro_f1.mean(),
            "std_macro_f1": group.macro_f1.std(ddof=0),
            "mean_balanced_accuracy": group.balanced_accuracy.mean(),
            "mean_weighted_f1": group.weighted_f1.mean(),
        }
        for label in LABELS:
            row[f"mean_recall_{label}"] = group[f"recall_{label}"].mean()
            row[f"mean_f1_{label}"] = group[f"f1_{label}"].mean()
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values("mean_macro_f1", ascending=False)
    write_csv(output / "frozen_embedding_aggregate.csv", aggregate)
    selected = aggregate.iloc[0].approach.replace("frozen_", "")
    selected_predictions = predictions[predictions.classifier == selected]
    matrix = pd.DataFrame(
        confusion_matrix(
            selected_predictions.severity, selected_predictions.predicted, labels=LABELS
        ),
        index=LABELS,
        columns=LABELS,
    )
    normalized = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    write_csv(output / "frozen_embedding_confusion_matrix.csv", matrix)
    write_csv(output / "frozen_embedding_confusion_matrix_normalized.csv", normalized)
    save_confusion_matrix_figure(
        output / "frozen_embedding_confusion_matrix.png",
        matrix,
        f"MYLYN frozen embeddings OOF: {selected}",
    )
    _error_analysis(frame, predictions, aggregate, output)
    model_info = {
        "model_identifier": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "cache_location": str((cache_root / "huggingface").resolve()),
        "embedding_cache": str(cache.resolve()),
        "embedding_dimension": embeddings.shape[1],
        "total_parameters": sum(
            parameter.numel()
            for parameter in _load_encoder(cache_root / "huggingface")[1].parameters()
        ),
        "trainable_encoder_parameters": 0,
        "max_sequence_length": MAX_LENGTH,
        "fold_hash": EXPECTED_FOLD_HASH,
        "material_threshold": MATERIAL_THRESHOLD,
        "material_threshold_met": bool(aggregate.iloc[0].mean_macro_f1 >= MATERIAL_THRESHOLD),
    }
    write_json(output / "model_resource_report.json", model_info)


def _error_analysis(
    frame: pd.DataFrame, predictions: pd.DataFrame, aggregate: pd.DataFrame, output: Path
) -> None:
    selected = aggregate.iloc[0].approach.replace("frozen_", "")
    chosen = predictions[predictions.classifier == selected].copy()
    chosen["correct"] = chosen.severity == chosen.predicted
    chosen["text_length"] = chosen.summary.str.len() + chosen.description.str.len()
    chosen["text_length_group"] = pd.cut(
        chosen.text_length, [-1, 200, 1000, np.inf], labels=["short", "medium", "long"]
    )
    errors = chosen[~chosen.correct]
    confusions = errors.groupby(["severity", "predicted"]).size().reset_index(name="errors")
    write_csv(output / "frozen_embedding_common_confusions.csv", confusions)
    by_length = (
        chosen.groupby("text_length_group", observed=True)
        .apply(
            lambda group: f1_score(
                group.severity, group.predicted, labels=LABELS, average="macro", zero_division=0
            ),
            include_groups=False,
        )
        .reset_index(name="macro_f1")
    )
    write_csv(output / "frozen_embedding_by_text_length.csv", by_length)
    examples = errors[errors.severity.isin(["blocker", "critical"])].head(100)
    write_csv(output / "blocker_critical_false_negatives.csv", examples)
    tfidf_path = output.parent / "mylyn" / "model_comparison.csv"
    tfidf = pd.read_csv(tfidf_path).iloc[0]
    comparison = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "approach": "tfidf_logistic",
                        "mean_macro_f1": tfidf.mean_cv_macro_f1,
                        "std_macro_f1": tfidf.std_cv_macro_f1,
                        **{
                            f"mean_recall_{label}": tfidf[f"mean_recall_{label}"]
                            for label in LABELS
                        },
                        **{f"mean_f1_{label}": tfidf[f"mean_f1_{label}"] for label in LABELS},
                    }
                ]
            ),
            aggregate,
        ],
        ignore_index=True,
    )
    write_csv(output / "tfidf_vs_pretrained.csv", comparison)
    per_class = []
    for _, row in comparison.iterrows():
        for label in LABELS:
            per_class.append(
                {
                    "approach": row.approach,
                    "severity": label,
                    "recall": row[f"mean_recall_{label}"],
                    "f1": row[f"mean_f1_{label}"],
                }
            )
    write_csv(output / "per_class_comparison.csv", pd.DataFrame(per_class))
    (output / "error_analysis.md").write_text(
        "# Development-only pretrained error analysis\n\n"
        f"Selected frozen classifier: {selected}. All predictions are fixed-fold OOF.\n"
        "Blocker/critical false negatives and representative text are redacted.\n"
        "TF-IDF comparison is aggregate because its row-level OOF predictions were not retained.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["hardware", "frozen"])
    parser.add_argument("--development", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/model_development/mylyn_pretrained")
    )
    parser.add_argument("--cache", type=Path, default=Path(".cache/mylyn_pretrained"))
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.command == "hardware":
        print(json.dumps(hardware_report(Path.cwd(), arguments.output), indent=2, default=str))
    else:
        if not arguments.development:
            parser.error("--development is required")
        run_frozen_embeddings(arguments.development, arguments.output, arguments.cache)
