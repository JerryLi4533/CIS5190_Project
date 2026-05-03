from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from train_model_search import (
    ARTIFACTS_DIR,
    DATA_PATH,
    LABELS,
    MODEL_PATH,
    PREPROCESS_PATH,
    WEIGHTS_PATH,
    ClassifierConfig,
    FeatureConfig,
    build_model_py,
    build_preprocess_py,
    fit_vectorizers,
    label_metrics,
    load_dataset,
    make_classifier,
    positive_coef_intercept,
    predict_with_threshold,
    safe_roc_auc,
    scores_for_positive_class,
    transform_texts,
    tune_threshold,
)

METRICS_PATH = ARTIFACTS_DIR / "word_trigram_ensemble_metrics.json"
RANDOM_STATE = 42


ENSEMBLES: dict[str, list[ClassifierConfig]] = {
    "svc_035_040_045": [
        ClassifierConfig("linearsvc", 0.35),
        ClassifierConfig("linearsvc", 0.40),
        ClassifierConfig("linearsvc", 0.45),
    ],
    "svc_040_045_050": [
        ClassifierConfig("linearsvc", 0.40),
        ClassifierConfig("linearsvc", 0.45),
        ClassifierConfig("linearsvc", 0.50),
    ],
    "svc_035_045_060": [
        ClassifierConfig("linearsvc", 0.35),
        ClassifierConfig("linearsvc", 0.45),
        ClassifierConfig("linearsvc", 0.60),
    ],
    "svc_040_045_050_060": [
        ClassifierConfig("linearsvc", 0.40),
        ClassifierConfig("linearsvc", 0.45),
        ClassifierConfig("linearsvc", 0.50),
        ClassifierConfig("linearsvc", 0.60),
    ],
    "svc_035_040_045_050_060_075": [
        ClassifierConfig("linearsvc", 0.35),
        ClassifierConfig("linearsvc", 0.40),
        ClassifierConfig("linearsvc", 0.45),
        ClassifierConfig("linearsvc", 0.50),
        ClassifierConfig("linearsvc", 0.60),
        ClassifierConfig("linearsvc", 0.75),
    ],
}


def word_trigram_feature_config(
    word_max_features: int,
    char_max_features: int,
) -> FeatureConfig:
    return FeatureConfig(
        "word_trigrams_ensemble",
        word_max_features,
        char_max_features,
        (1, 3),
        (3, 5),
        min_df=1,
        sublinear_tf=True,
    )


def average_scores_for_configs(
    X_train,
    train_labels: list[str],
    X_validation,
    classifier_configs: list[ClassifierConfig],
) -> np.ndarray:
    scores = []
    for classifier_config in classifier_configs:
        clf = make_classifier(classifier_config)
        clf.fit(X_train, train_labels)
        scores.append(scores_for_positive_class(clf, X_validation))
    return np.mean(np.vstack(scores), axis=0)


def summarize_predictions(
    labels: list[str],
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    default_preds = predict_with_threshold(scores, 0.0)
    threshold_preds = predict_with_threshold(scores, threshold)
    default_metrics = label_metrics(labels, default_preds)
    threshold_metrics = label_metrics(labels, threshold_preds)

    return {
        "default_accuracy": default_metrics["accuracy"],
        "threshold_accuracy": threshold_metrics["accuracy"],
        "default_macro_f1": default_metrics["macro_f1"],
        "threshold_macro_f1": threshold_metrics["macro_f1"],
        "default_weighted_f1": default_metrics["weighted_f1"],
        "threshold_weighted_f1": threshold_metrics["weighted_f1"],
        "roc_auc": safe_roc_auc(labels, scores),
        "threshold": float(threshold),
        "classification_report": classification_report(
            labels,
            threshold_preds,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(labels, threshold_preds, labels=LABELS).tolist(),
    }


def evaluate_ensemble_cv(
    texts: list[str],
    labels: list[str],
    feature_config: FeatureConfig,
    ensemble_name: str,
    classifier_configs: list[ClassifierConfig],
    folds: int,
) -> dict[str, Any]:
    labels_array = np.asarray(labels)
    oof_scores = np.zeros(len(labels), dtype=float)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    for train_idx, validation_idx in skf.split(texts, labels):
        train_texts = [texts[index] for index in train_idx]
        validation_texts = [texts[index] for index in validation_idx]
        train_labels = labels_array[train_idx].tolist()

        word_vectorizer, char_vectorizer = fit_vectorizers(train_texts, feature_config)
        X_train = transform_texts(train_texts, word_vectorizer, char_vectorizer)
        X_validation = transform_texts(validation_texts, word_vectorizer, char_vectorizer)
        oof_scores[validation_idx] = average_scores_for_configs(
            X_train,
            train_labels,
            X_validation,
            classifier_configs,
        )

    threshold, _ = tune_threshold(oof_scores, labels)
    summary = summarize_predictions(labels, oof_scores, threshold)
    summary.update(
        {
            "ensemble_name": ensemble_name,
            "validation_protocol": "stratified_cv",
            "feature_config": {
                "word_max_features": feature_config.word_max_features,
                "char_max_features": feature_config.char_max_features,
                "word_ngram_range": list(feature_config.word_ngram_range),
                "char_ngram_range": list(feature_config.char_ngram_range),
                "min_df": feature_config.min_df,
                "sublinear_tf": feature_config.sublinear_tf,
            },
            "classifier_configs": [
                {"kind": config.kind, "c": config.c, "class_weight": config.class_weight}
                for config in classifier_configs
            ],
        }
    )
    return summary


def evaluate_ensemble_holdout(
    train_texts: list[str],
    train_labels: list[str],
    validation_texts: list[str],
    validation_labels: list[str],
    feature_config: FeatureConfig,
    ensemble_name: str,
    classifier_configs: list[ClassifierConfig],
) -> dict[str, Any]:
    word_vectorizer, char_vectorizer = fit_vectorizers(train_texts, feature_config)
    X_train = transform_texts(train_texts, word_vectorizer, char_vectorizer)
    X_validation = transform_texts(validation_texts, word_vectorizer, char_vectorizer)
    scores = average_scores_for_configs(
        X_train,
        train_labels,
        X_validation,
        classifier_configs,
    )
    threshold, _ = tune_threshold(scores, validation_labels)
    summary = summarize_predictions(validation_labels, scores, threshold)
    summary.update(
        {
            "ensemble_name": ensemble_name,
            "validation_protocol": "holdout_csv",
            "feature_config": {
                "word_max_features": feature_config.word_max_features,
                "char_max_features": feature_config.char_max_features,
                "word_ngram_range": list(feature_config.word_ngram_range),
                "char_ngram_range": list(feature_config.char_ngram_range),
                "min_df": feature_config.min_df,
                "sublinear_tf": feature_config.sublinear_tf,
            },
            "classifier_configs": [
                {"kind": config.kind, "c": config.c, "class_weight": config.class_weight}
                for config in classifier_configs
            ],
        }
    )
    return summary


def save_averaged_linear_state_dict(
    coefs: list[np.ndarray],
    intercepts: list[float],
    threshold: float,
    num_features: int,
) -> None:
    import torch

    coef = np.mean(np.vstack(coefs), axis=0)
    intercept = float(np.mean(intercepts))
    weights = torch.zeros((len(LABELS), num_features), dtype=torch.float32)
    bias = torch.zeros(len(LABELS), dtype=torch.float32)
    positive_index = LABELS.index(LABELS[1])
    weights[positive_index] = torch.tensor(coef, dtype=torch.float32)
    bias[positive_index] = torch.tensor(intercept - threshold, dtype=torch.float32)
    torch.save({"linear.weight": weights, "linear.bias": bias}, WEIGHTS_PATH)


def fit_final_and_export_ensemble(
    texts: list[str],
    labels: list[str],
    feature_config: FeatureConfig,
    classifier_configs: list[ClassifierConfig],
    threshold: float,
    export: bool,
) -> dict[str, Any]:
    word_vectorizer, char_vectorizer = fit_vectorizers(texts, feature_config)
    X_all = transform_texts(texts, word_vectorizer, char_vectorizer)

    coefs = []
    intercepts = []
    for classifier_config in classifier_configs:
        clf = make_classifier(classifier_config)
        clf.fit(X_all, labels)
        coef, intercept = positive_coef_intercept(clf)
        coefs.append(coef)
        intercepts.append(intercept)

    num_features = int(X_all.shape[1])
    if export:
        save_averaged_linear_state_dict(coefs, intercepts, threshold, num_features)
        MODEL_PATH.write_text(build_model_py(num_features), encoding="utf-8")
        PREPROCESS_PATH.write_text(
            build_preprocess_py(word_vectorizer, char_vectorizer, feature_config),
            encoding="utf-8",
        )

    return {
        "num_features": num_features,
        "threshold_exported_as_bias_shift": threshold,
        "weights_path": str(WEIGHTS_PATH),
        "model_path": str(MODEL_PATH),
        "preprocess_path": str(PREPROCESS_PATH),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train same-feature LinearSVC ensembles and export the best as one linear model."
    )
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument("--validation-csv", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--word-max-features", type=int, default=20000)
    parser.add_argument("--char-max-features", type=int, default=30000)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    texts, labels, dataset_summary = load_dataset(args.csv)
    feature_config = word_trigram_feature_config(args.word_max_features, args.char_max_features)

    validation_texts = None
    validation_labels = None
    validation_dataset_summary = None
    if args.validation_csv is not None:
        validation_texts, validation_labels, validation_dataset_summary = load_dataset(args.validation_csv)

    results = []
    for ensemble_name, classifier_configs in ENSEMBLES.items():
        print(f"[ensemble] {ensemble_name}")
        if validation_texts is not None and validation_labels is not None:
            summary = evaluate_ensemble_holdout(
                texts,
                labels,
                validation_texts,
                validation_labels,
                feature_config,
                ensemble_name,
                classifier_configs,
            )
        else:
            summary = evaluate_ensemble_cv(
                texts,
                labels,
                feature_config,
                ensemble_name,
                classifier_configs,
                args.folds,
            )
        results.append(summary)
        print(
            "    "
            f"accuracy={summary['threshold_accuracy']:.4f} "
            f"macro_f1={summary['threshold_macro_f1']:.4f} "
            f"threshold={summary['threshold']:.4f}"
        )

    results.sort(
        key=lambda result: (
            result["threshold_accuracy"],
            result["threshold_macro_f1"],
        ),
        reverse=True,
    )
    best = results[0]
    export_summary = fit_final_and_export_ensemble(
        texts,
        labels,
        feature_config,
        ENSEMBLES[best["ensemble_name"]],
        best["threshold"],
        export=not args.no_export,
    )

    metrics = {
        "dataset_path": str(args.csv),
        "dataset": dataset_summary,
        "validation_dataset_path": str(args.validation_csv) if args.validation_csv else None,
        "validation_dataset": validation_dataset_summary,
        "selection_metric": "threshold_accuracy",
        "best_model": best,
        "final_export": export_summary,
        "all_results": results,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nBest ensemble")
    print(json.dumps(best, indent=2))
    if args.no_export:
        print(f"\nsaved metrics to {METRICS_PATH}; export skipped")
    else:
        print(f"\nsaved averaged ensemble weights to {WEIGHTS_PATH}")
        print(f"rewrote model constants in {MODEL_PATH}")
        print(f"rewrote preprocess constants in {PREPROCESS_PATH}")
        print(f"saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
