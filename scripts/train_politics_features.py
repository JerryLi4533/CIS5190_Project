from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from analyze_errors import (
    ARTIFACTS_DIR,
    CHAR_MAX_FEATURES,
    CHAR_NGRAM_RANGE,
    ENSEMBLE_C_VALUES,
    FOLDS,
    LABELS,
    RANDOM_STATE,
    WORD_MAX_FEATURES,
    WORD_NGRAM_RANGE,
    add_analysis_columns,
    headline_meta_features,
    load_raw_clean_dataset,
)
from train_model_search import DATA_PATH, predict_with_threshold, tune_threshold


METRICS_PATH = ARTIFACTS_DIR / "politics_features_metrics.json"
SUMMARY_PATH = ARTIFACTS_DIR / "politics_features_summary.md"
PREDICTIONS_PATH = ARTIFACTS_DIR / "politics_features_predictions.csv"
ERRORS_PATH = ARTIFACTS_DIR / "politics_features_errors.csv"

POLITICAL_PEOPLE = {
    "trump",
    "biden",
    "harris",
    "vance",
    "walz",
    "pence",
    "obama",
    "clinton",
    "schumer",
    "pelosi",
    "mcconnell",
    "desantis",
    "zelenskyy",
}
POLITICAL_INSTITUTIONS = {
    "white house",
    "congress",
    "senate",
    "house",
    "supreme court",
    "doj",
    "fbi",
    "rnc",
    "dnc",
}
PARTY_TERMS = {
    "gop",
    "republican",
    "republicans",
    "democrat",
    "democrats",
    "dem",
    "dems",
    "liberal",
    "conservative",
    "progressive",
}
ELECTION_TERMS = {
    "election",
    "campaign",
    "poll",
    "polls",
    "voters",
    "voting",
    "ballot",
    "primary",
    "nominee",
    "debate",
    "rally",
    "swing state",
}
FRAMING_VERBS = {
    "says",
    "said",
    "warns",
    "slams",
    "blasts",
    "rips",
    "claims",
    "accuses",
    "admits",
    "calls",
    "urges",
    "defends",
    "condemns",
    "attacks",
    "criticizes",
    "mocks",
}
POLICY_TERMS = {
    "border",
    "immigration",
    "abortion",
    "crime",
    "tax",
    "tariff",
    "inflation",
    "economy",
    "israel",
    "gaza",
    "ukraine",
    "russia",
    "china",
    "health care",
    "gun",
    "guns",
}


def count_terms(text: str, terms: set[str]) -> float:
    return float(sum(text.count(term) for term in terms))


def contains_terms(text: str, terms: set[str]) -> float:
    return float(any(term in text for term in terms))


def enhanced_meta_features(texts: list[str]) -> np.ndarray:
    base = headline_meta_features(texts)
    rows: list[list[float]] = []
    for text in texts:
        raw = str(text)
        lower = raw.lower()
        words = re.findall(r"\b\w+\b", lower)
        word_count = len(words)
        person_count = count_terms(lower, POLITICAL_PEOPLE)
        party_count = count_terms(lower, PARTY_TERMS)
        election_count = count_terms(lower, ELECTION_TERMS)
        framing_count = count_terms(lower, FRAMING_VERBS)
        policy_count = count_terms(lower, POLICY_TERMS)
        institution_count = count_terms(lower, POLITICAL_INSTITUTIONS)
        person_plus_says = float(
            bool(
                re.search(
                    r"\b(?:trump|biden|harris|vance|walz|pence|obama|clinton)\b.{0,30}\b(?:says|said|warns|claims|calls|urges)\b",
                    lower,
                )
            )
        )
        quote_after_colon = float(bool(re.search(r":\s*['\"]", raw)))
        starts_with_person = float(
            bool(
                re.match(
                    r"^(?:Donald\s+Trump|Joe\s+Biden|Kamala\s+Harris|JD\s+Vance|Tim\s+Walz|Mike\s+Pence|Trump|Biden|Harris)\b",
                    raw,
                    flags=re.IGNORECASE,
                )
            )
        )
        poll_headline = float(bool(re.search(r"\bpoll(?:s|ing)?\b", lower)))
        campaign_says = float(bool(re.search(r"\bcampaign\s+(?:says|said|claims|warns)\b", lower)))
        white_house_says = float(bool(re.search(r"\bwhite house\s+(?:says|said|warns|condemns)\b", lower)))
        opinion_framing = float(
            bool(
                re.search(
                    r"\b(?:opinion|exclusive|fact check|analysis|power rankings|town hall)\b",
                    lower,
                )
            )
        )
        rows.append(
            [
                contains_terms(lower, POLITICAL_PEOPLE),
                person_count / max(word_count, 1),
                contains_terms(lower, PARTY_TERMS),
                party_count / max(word_count, 1),
                contains_terms(lower, ELECTION_TERMS),
                election_count / max(word_count, 1),
                contains_terms(lower, FRAMING_VERBS),
                framing_count / max(word_count, 1),
                contains_terms(lower, POLICY_TERMS),
                policy_count / max(word_count, 1),
                contains_terms(lower, POLITICAL_INSTITUTIONS),
                institution_count / max(word_count, 1),
                person_plus_says,
                quote_after_colon,
                starts_with_person,
                poll_headline,
                campaign_says,
                white_house_says,
                opinion_framing,
                float("?" in raw and contains_terms(lower, POLITICAL_PEOPLE)),
            ]
        )
    return np.hstack([base, np.asarray(rows, dtype=float)])


def make_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=WORD_MAX_FEATURES,
        ngram_range=WORD_NGRAM_RANGE,
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w\w+\b",
        dtype=np.float32,
    )


def make_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        max_features=CHAR_MAX_FEATURES,
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )


def fit_features(
    clean_texts: list[str],
    raw_texts: list[str],
    meta_fn: Callable[[list[str]], np.ndarray],
):
    word_vectorizer = make_word_vectorizer()
    char_vectorizer = make_char_vectorizer()
    meta_scaler = StandardScaler(with_mean=False)
    X = hstack(
        [
            word_vectorizer.fit_transform(clean_texts),
            char_vectorizer.fit_transform(clean_texts),
            csr_matrix(meta_scaler.fit_transform(meta_fn(raw_texts))),
        ],
        format="csr",
    )
    return word_vectorizer, char_vectorizer, meta_scaler, X


def transform_features(
    clean_texts: list[str],
    raw_texts: list[str],
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    meta_scaler: StandardScaler,
    meta_fn: Callable[[list[str]], np.ndarray],
):
    return hstack(
        [
            word_vectorizer.transform(clean_texts),
            char_vectorizer.transform(clean_texts),
            csr_matrix(meta_scaler.transform(meta_fn(raw_texts))),
        ],
        format="csr",
    )


def positive_scores(clf: LinearSVC, X) -> np.ndarray:
    scores = np.asarray(clf.decision_function(X), dtype=float)
    classes = [str(item) for item in clf.classes_]
    return scores if classes[1] == LABELS[1] else -scores


def average_scores(X_train, train_labels: list[str], X_validation) -> np.ndarray:
    scores = []
    for c_value in ENSEMBLE_C_VALUES:
        clf = LinearSVC(
            C=c_value,
            class_weight="balanced",
            max_iter=20000,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, train_labels)
        scores.append(positive_scores(clf, X_validation))
    return np.mean(np.vstack(scores), axis=0)


def evaluate_oof(frame: pd.DataFrame, meta_fn: Callable[[list[str]], np.ndarray]) -> tuple[dict[str, Any], pd.DataFrame]:
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

        word_vectorizer, char_vectorizer, meta_scaler, X_train = fit_features(
            train_clean,
            train_raw,
            meta_fn,
        )
        X_validation = transform_features(
            validation_clean,
            validation_raw,
            word_vectorizer,
            char_vectorizer,
            meta_scaler,
            meta_fn,
        )
        oof_scores[validation_idx] = average_scores(X_train, train_labels, X_validation)
        fold_ids[validation_idx] = fold

    threshold, _ = tune_threshold(oof_scores, labels)
    preds = predict_with_threshold(oof_scores, threshold)
    binary_targets = np.asarray([1 if label == LABELS[1] else 0 for label in labels], dtype=int)
    predictions = frame.copy()
    predictions["fold"] = fold_ids
    predictions["score_nbc"] = oof_scores
    predictions["predicted_label"] = preds
    predictions["correct"] = predictions["label"] == predictions["predicted_label"]
    predictions["error_direction"] = predictions["label"] + "->" + predictions["predicted_label"]
    predictions.loc[predictions["correct"], "error_direction"] = ""
    predictions = add_analysis_columns(predictions, threshold)

    summary = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=LABELS, average="macro")),
        "roc_auc": float(roc_auc_score(binary_targets, oof_scores)),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(labels, preds, labels=LABELS).tolist(),
        "num_errors": int((~predictions["correct"]).sum()),
    }
    return summary, predictions


def subset_summary(predictions: pd.DataFrame, subset: pd.Series) -> dict[str, Any]:
    part = predictions.loc[subset].copy()
    labels = part["label"].tolist()
    preds = part["predicted_label"].tolist()
    binary_targets = np.asarray([1 if label == LABELS[1] else 0 for label in labels], dtype=int)
    return {
        "examples": int(len(part)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=LABELS, average="macro")),
        "roc_auc": float(roc_auc_score(binary_targets, part["score_nbc"].to_numpy(dtype=float))),
        "num_errors": int((~part["correct"]).sum()),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(labels, preds, labels=LABELS).tolist(),
    }


def build_summary(metrics: dict[str, Any]) -> str:
    baseline = metrics["baseline_10_meta"]
    enhanced = metrics["enhanced_politics_meta"]
    base_pol = metrics["baseline_politics_subset"]
    enhanced_pol = metrics["enhanced_politics_subset"]
    matrix = enhanced["confusion_matrix"]
    top_errors = metrics["top_enhanced_errors"][:12]
    error_lines = "\n".join(
        f"- true={item['label']}, pred={item['predicted_label']}, margin={item['confidence_margin']:.3f}: {item['headline']}"
        for item in top_errors
    )
    if not error_lines:
        error_lines = "- None"

    return f"""# Politics Feature Experiment Summary

## Overall OOF Results

| Model | Accuracy | Macro F1 | ROC-AUC | Errors | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 10 meta + mojibake repair | {baseline["accuracy"]:.4f} | {baseline["macro_f1"]:.4f} | {baseline["roc_auc"]:.4f} | {baseline["num_errors"]} | {baseline["threshold"]:.6f} |
| Enhanced politics meta | {enhanced["accuracy"]:.4f} | {enhanced["macro_f1"]:.4f} | {enhanced["roc_auc"]:.4f} | {enhanced["num_errors"]} | {enhanced["threshold"]:.6f} |

## Politics Subset Results

| Model | Politics Accuracy | Macro F1 | ROC-AUC | Errors |
| --- | ---: | ---: | ---: | ---: |
| Baseline 10 meta + mojibake repair | {base_pol["accuracy"]:.4f} | {base_pol["macro_f1"]:.4f} | {base_pol["roc_auc"]:.4f} | {base_pol["num_errors"]} |
| Enhanced politics meta | {enhanced_pol["accuracy"]:.4f} | {enhanced_pol["macro_f1"]:.4f} | {enhanced_pol["roc_auc"]:.4f} | {enhanced_pol["num_errors"]} |

## Enhanced Overall Confusion Matrix

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | {matrix[0][0]} | {matrix[0][1]} |
| NBC | {matrix[1][0]} | {matrix[1][1]} |

## Highest-Confidence Enhanced Errors

{error_lines}
"""


def run_politics_features(data_path: Path) -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    frame, dataset_summary = load_raw_clean_dataset(data_path)
    baseline_summary, baseline_predictions = evaluate_oof(frame, headline_meta_features)
    enhanced_summary, enhanced_predictions = evaluate_oof(frame, enhanced_meta_features)

    politics_mask = enhanced_predictions["topic"] == "politics"
    enhanced_errors = enhanced_predictions[~enhanced_predictions["correct"]].copy()
    enhanced_errors = enhanced_errors.sort_values("confidence_margin", ascending=False)

    metrics = {
        "dataset_path": str(data_path),
        "dataset": dataset_summary,
        "model": "enhanced_politics_features_experiment",
        "classifier_c_values": list(ENSEMBLE_C_VALUES),
        "feature_config": {
            "word_max_features": WORD_MAX_FEATURES,
            "char_max_features": CHAR_MAX_FEATURES,
            "baseline_meta_features": int(headline_meta_features(["example headline"]).shape[1]),
            "enhanced_meta_features": int(enhanced_meta_features(["example headline"]).shape[1]),
            "word_ngram_range": list(WORD_NGRAM_RANGE),
            "char_ngram_range": list(CHAR_NGRAM_RANGE),
            "text_mode": "headline",
            "mojibake_repair": True,
        },
        "baseline_10_meta": baseline_summary,
        "enhanced_politics_meta": enhanced_summary,
        "baseline_politics_subset": subset_summary(baseline_predictions, politics_mask),
        "enhanced_politics_subset": subset_summary(enhanced_predictions, politics_mask),
        "top_enhanced_errors": enhanced_errors[
            ["label", "predicted_label", "confidence_margin", "topic", "headline"]
        ]
        .head(30)
        .to_dict(orient="records"),
    }

    enhanced_predictions.to_csv(PREDICTIONS_PATH, index=False)
    enhanced_errors.to_csv(ERRORS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(build_summary(metrics), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    metrics = run_politics_features(args.data)
    print(json.dumps(metrics, indent=2))
    print(f"saved metrics to {METRICS_PATH}")
    print(f"saved summary to {SUMMARY_PATH}")
    print(f"saved predictions to {PREDICTIONS_PATH}")
    print(f"saved errors to {ERRORS_PATH}")


if __name__ == "__main__":
    main()
