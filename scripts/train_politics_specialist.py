from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from analyze_errors import (
    ARTIFACTS_DIR,
    ENSEMBLE_C_VALUES,
    FOLDS,
    LABELS,
    RANDOM_STATE,
    add_analysis_columns,
    average_scores,
    fit_features,
    headline_meta_features,
    load_raw_clean_dataset,
    transform_features,
)
from train_model_search import DATA_PATH, predict_with_threshold, tune_threshold


METRICS_PATH = ARTIFACTS_DIR / "politics_specialist_metrics.json"
SUMMARY_PATH = ARTIFACTS_DIR / "politics_specialist_summary.md"
PREDICTIONS_PATH = ARTIFACTS_DIR / "politics_specialist_predictions.csv"
ERRORS_PATH = ARTIFACTS_DIR / "politics_specialist_errors.csv"


def oof_scores_for_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels = frame["label"].tolist()
    clean_texts = frame["clean_headline"].tolist()
    raw_texts = frame["headline"].tolist()
    labels_array = np.asarray(labels)
    oof_scores = np.zeros(len(frame), dtype=float)
    fold_ids = np.zeros(len(frame), dtype=int)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, validation_idx) in enumerate(skf.split(clean_texts, labels), start=1):
        train_clean = [clean_texts[index] for index in train_idx]
        validation_clean = [clean_texts[index] for index in validation_idx]
        train_raw = [raw_texts[index] for index in train_idx]
        validation_raw = [raw_texts[index] for index in validation_idx]
        train_labels = labels_array[train_idx].tolist()

        word_vectorizer, char_vectorizer, meta_scaler, X_train = fit_features(train_clean, train_raw)
        X_validation = transform_features(
            validation_clean,
            validation_raw,
            word_vectorizer,
            char_vectorizer,
            meta_scaler,
        )
        oof_scores[validation_idx] = average_scores(X_train, train_labels, X_validation)
        fold_ids[validation_idx] = fold

    return oof_scores, fold_ids


def summarize_predictions(labels: list[str], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = predict_with_threshold(scores, threshold)
    binary_targets = np.asarray([1 if label == LABELS[1] else 0 for label in labels], dtype=int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=LABELS, average="macro")),
        "roc_auc": float(roc_auc_score(binary_targets, scores)),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(labels, preds, labels=LABELS).tolist(),
        "predictions": preds,
    }


def build_summary(metrics: dict[str, Any]) -> str:
    general = metrics["general_model_on_politics"]
    specialist = metrics["politics_specialist"]
    routed = metrics["routed_general_plus_politics_specialist"]
    specialist_matrix = specialist["confusion_matrix"]
    top_errors = metrics["top_politics_specialist_errors"][:12]
    error_lines = "\n".join(
        f"- true={item['label']}, pred={item['predicted_label']}, margin={item['confidence_margin']:.3f}: {item['headline']}"
        for item in top_errors
    )
    if not error_lines:
        error_lines = "- None"

    return f"""# Politics Specialist Summary

## Dataset

- Politics examples: {metrics["politics_examples"]}
- FoxNews politics examples: {metrics["politics_label_counts"].get("FoxNews", 0)}
- NBC politics examples: {metrics["politics_label_counts"].get("NBC", 0)}
- Mojibake repair: enabled for this experiment

## Accuracy Comparison

| Model | Scope | Accuracy | Macro F1 | ROC-AUC | Threshold |
| --- | --- | ---: | ---: | ---: | ---: |
| Current-style general OOF | politics only | {general["accuracy"]:.4f} | {general["macro_f1"]:.4f} | {general["roc_auc"]:.4f} | {general["threshold"]:.6f} |
| Politics specialist OOF | politics only | {specialist["accuracy"]:.4f} | {specialist["macro_f1"]:.4f} | {specialist["roc_auc"]:.4f} | {specialist["threshold"]:.6f} |
| Routed OOF | all examples | {routed["accuracy"]:.4f} | {routed["macro_f1"]:.4f} | {routed["roc_auc"]:.4f} | mixed |

## Politics Specialist Confusion Matrix

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | {specialist_matrix[0][0]} | {specialist_matrix[0][1]} |
| NBC | {specialist_matrix[1][0]} | {specialist_matrix[1][1]} |

## Highest-Confidence Politics Specialist Errors

{error_lines}
"""


def run_politics_specialist(data_path: Path) -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    frame, dataset_summary = load_raw_clean_dataset(data_path)

    general_scores, general_fold_ids = oof_scores_for_frame(frame)
    general_threshold, _ = tune_threshold(general_scores, frame["label"].tolist())
    general_predictions = frame.copy()
    general_predictions["fold"] = general_fold_ids
    general_predictions["score_nbc"] = general_scores
    general_predictions["predicted_label"] = predict_with_threshold(general_scores, general_threshold)
    general_predictions["correct"] = (
        general_predictions["label"] == general_predictions["predicted_label"]
    )
    general_predictions["error_direction"] = (
        general_predictions["label"] + "->" + general_predictions["predicted_label"]
    )
    general_predictions.loc[general_predictions["correct"], "error_direction"] = ""
    general_predictions = add_analysis_columns(general_predictions, general_threshold)

    politics_mask = general_predictions["topic"] == "politics"
    politics_frame = frame.loc[politics_mask].reset_index(drop=True)
    politics_general = general_predictions.loc[politics_mask].copy()
    politics_labels = politics_frame["label"].tolist()

    politics_scores, politics_fold_ids = oof_scores_for_frame(politics_frame)
    politics_threshold, _ = tune_threshold(politics_scores, politics_labels)
    politics_summary = summarize_predictions(politics_labels, politics_scores, politics_threshold)

    general_politics_summary = summarize_predictions(
        politics_labels,
        politics_general["score_nbc"].to_numpy(dtype=float),
        general_threshold,
    )

    specialist_predictions = politics_frame.copy()
    specialist_predictions["fold"] = politics_fold_ids
    specialist_predictions["score_nbc"] = politics_scores
    specialist_predictions["predicted_label"] = politics_summary["predictions"]
    specialist_predictions["correct"] = (
        specialist_predictions["label"] == specialist_predictions["predicted_label"]
    )
    specialist_predictions["error_direction"] = (
        specialist_predictions["label"] + "->" + specialist_predictions["predicted_label"]
    )
    specialist_predictions.loc[specialist_predictions["correct"], "error_direction"] = ""
    specialist_predictions = add_analysis_columns(specialist_predictions, politics_threshold)
    specialist_errors = specialist_predictions[~specialist_predictions["correct"]].copy()
    specialist_errors = specialist_errors.sort_values("confidence_margin", ascending=False)

    routed_scores = general_scores.copy()
    routed_preds = np.asarray(predict_with_threshold(general_scores, general_threshold), dtype=object)
    routed_preds[politics_mask.to_numpy()] = politics_summary["predictions"]
    routed_labels = frame["label"].tolist()
    routed_binary_targets = np.asarray(
        [1 if label == LABELS[1] else 0 for label in routed_labels],
        dtype=int,
    )
    routed_scores[politics_mask.to_numpy()] = politics_scores
    routed_summary = {
        "accuracy": float(accuracy_score(routed_labels, routed_preds.tolist())),
        "macro_f1": float(f1_score(routed_labels, routed_preds.tolist(), labels=LABELS, average="macro")),
        "roc_auc": float(roc_auc_score(routed_binary_targets, routed_scores)),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(routed_labels, routed_preds.tolist(), labels=LABELS).tolist(),
    }

    metrics = {
        "dataset_path": str(data_path),
        "dataset": dataset_summary,
        "model": "politics_specialist_experiment",
        "classifier_c_values": list(ENSEMBLE_C_VALUES),
        "feature_config": {
            "word_max_features": 30000,
            "char_max_features": 50000,
            "meta_features": int(headline_meta_features(["example headline"]).shape[1]),
            "word_ngram_range": [1, 3],
            "char_ngram_range": [2, 5],
            "text_mode": "headline",
        },
        "politics_examples": int(len(politics_frame)),
        "politics_label_counts": politics_frame["label"].value_counts().to_dict(),
        "general_model_on_politics": {
            key: value
            for key, value in general_politics_summary.items()
            if key != "predictions"
        },
        "politics_specialist": {
            key: value
            for key, value in politics_summary.items()
            if key != "predictions"
        },
        "routed_general_plus_politics_specialist": routed_summary,
        "top_politics_specialist_errors": specialist_errors[
            ["label", "predicted_label", "confidence_margin", "headline"]
        ]
        .head(30)
        .to_dict(orient="records"),
    }

    specialist_predictions.to_csv(PREDICTIONS_PATH, index=False)
    specialist_errors.to_csv(ERRORS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(build_summary(metrics), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    metrics = run_politics_specialist(args.data)
    print(json.dumps(metrics, indent=2))
    print(f"saved metrics to {METRICS_PATH}")
    print(f"saved summary to {SUMMARY_PATH}")
    print(f"saved predictions to {PREDICTIONS_PATH}")
    print(f"saved errors to {ERRORS_PATH}")


if __name__ == "__main__":
    main()
