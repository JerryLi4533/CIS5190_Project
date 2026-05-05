from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from train_model_search import (
    ARTIFACTS_DIR,
    DATA_PATH,
    LABELS,
    RANDOM_STATE,
    build_model_text,
    infer_label,
    predict_with_threshold,
    select_headlines,
    tune_threshold,
)


WORD_MAX_FEATURES = 30000
CHAR_MAX_FEATURES = 50000
WORD_NGRAM_RANGE = (1, 3)
CHAR_NGRAM_RANGE = (2, 5)
ENSEMBLE_C_VALUES = (0.8, 1.0)
FOLDS = 5

PREDICTIONS_PATH = ARTIFACTS_DIR / "error_analysis_predictions.csv"
ERRORS_PATH = ARTIFACTS_DIR / "error_analysis_errors.csv"
HIGH_CONFIDENCE_ERRORS_PATH = ARTIFACTS_DIR / "error_analysis_high_confidence_errors.csv"
SUMMARY_PATH = ARTIFACTS_DIR / "error_analysis_summary.md"
METRICS_PATH = ARTIFACTS_DIR / "error_analysis_metrics.json"


SOURCE_SUFFIX_PATTERN = re.compile(
    r"\s+[-|:]\s+"
    r"(fox news|fox business|nbc news|nbc news now|nbc select|msnbc|today)"
    r"\s*$",
    flags=re.IGNORECASE,
)

TOPIC_PATTERNS = {
    "politics": r"\b(?:trump|biden|white house|congress|senate|house|gop|republican|democrat|election|campaign|border|court|judge|lawmakers)\b",
    "crime": r"\b(?:police|killed|dead|death|murder|shooting|arrest|charged|crime|suspect|trial|prison|jail)\b",
    "world": r"\b(?:china|russia|ukraine|israel|gaza|iran|mexico|europe|foreign|war|military)\b",
    "health": r"\b(?:health|doctor|medical|covid|vaccine|study|cancer|disease|drug|hospital)\b",
    "business": r"\b(?:stock|market|business|company|ceo|economy|inflation|money|tax|price|bank)\b",
    "culture": r"\b(?:movie|music|celebrity|star|tv|hollywood|sports|nfl|nba|game|season)\b",
}

MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "Â\xa0": " ",
    "Â": "",
    "鈥檚": "'s",
    "鈥檛": "n't",
    "鈥": "'",
    "閳ユ獨": "'s",
}


def repair_mojibake(text: str) -> str:
    repaired = str(text)
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(bad, good)
    return repaired


def strip_source_suffix(text: str) -> str:
    text = repair_mojibake(text).strip()
    text = SOURCE_SUFFIX_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def headline_meta_features(texts: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for text in texts:
        raw = strip_source_suffix(text)
        chars = len(raw)
        words = re.findall(r"\b\w+\b", raw)
        nwords = len(words)
        letters = sum(1 for char in raw if char.isalpha())
        uppercase = sum(1 for char in raw if char.isupper())
        punctuation = sum(1 for char in raw if char in "!?;:'\".,-$%&()[]")
        rows.append(
            [
                math.log1p(chars),
                math.log1p(nwords),
                uppercase / max(letters, 1),
                punctuation / max(chars, 1),
                raw.count("?"),
                raw.count("!"),
                raw.count("'"),
                raw.count(":"),
                raw.count(","),
                float(any(char.isdigit() for char in raw)),
            ]
        )
    return np.asarray(rows, dtype=float)


def load_raw_clean_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    if "url" in df.columns:
        labels = [infer_label(url) for url in df["url"].tolist()]
    elif "label" in df.columns:
        labels = df["label"].astype(str).tolist()
    elif "source" in df.columns:
        labels = df["source"].astype(str).tolist()
    else:
        raise ValueError("Expected one of: url, label, source columns.")

    raw_headlines = [repair_mojibake(headline) for headline in select_headlines(df, labels)]
    urls = df["url"].fillna("").astype(str).tolist() if "url" in df.columns else [""] * len(df)
    work = pd.DataFrame(
        {
            "source_row": np.arange(len(df)),
            "url": urls,
            "headline": raw_headlines,
            "clean_headline": [
                build_model_text(headline, url, text_mode="headline")
                for headline, url in zip(raw_headlines, urls)
            ],
            "label": labels,
        }
    )

    input_examples = len(work)
    blank_examples = int((work["clean_headline"] == "").sum())
    work = work[work["clean_headline"] != ""].copy()

    conflicting = (
        work.groupby("clean_headline")["label"]
        .nunique()
        .loc[lambda series: series > 1]
        .index
    )
    conflict_examples = int(work["clean_headline"].isin(conflicting).sum())
    if len(conflicting) > 0:
        work = work[~work["clean_headline"].isin(conflicting)].copy()

    duplicate_examples = int(work.duplicated("clean_headline").sum())
    work = work.drop_duplicates("clean_headline", keep="first").reset_index(drop=True)

    summary = {
        "input_examples": input_examples,
        "used_examples": int(len(work)),
        "blank_examples": blank_examples,
        "conflicting_duplicate_examples": conflict_examples,
        "duplicate_headline_examples": duplicate_examples,
        "label_counts": work["label"].value_counts().to_dict(),
    }
    return work, summary


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


def fit_features(clean_texts: list[str], raw_texts: list[str]):
    word_vectorizer = make_word_vectorizer()
    char_vectorizer = make_char_vectorizer()
    meta_scaler = StandardScaler(with_mean=False)
    X = hstack(
        [
            word_vectorizer.fit_transform(clean_texts),
            char_vectorizer.fit_transform(clean_texts),
            csr_matrix(meta_scaler.fit_transform(headline_meta_features(raw_texts))),
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
):
    return hstack(
        [
            word_vectorizer.transform(clean_texts),
            char_vectorizer.transform(clean_texts),
            csr_matrix(meta_scaler.transform(headline_meta_features(raw_texts))),
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


def add_analysis_columns(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = frame.copy()
    out["margin_from_threshold"] = out["score_nbc"] - threshold
    out["confidence_margin"] = out["margin_from_threshold"].abs()
    out["headline_length"] = out["headline"].astype(str).str.len()
    out["word_count"] = out["headline"].astype(str).str.findall(r"\b\w+\b").str.len()
    out["has_quote"] = out["headline"].astype(str).str.contains("\"|'", regex=True)
    out["has_question"] = out["headline"].astype(str).str.contains(r"\?", regex=True)
    out["has_digit"] = out["headline"].astype(str).str.contains(r"\d", regex=True)
    lower = out["clean_headline"].astype(str).str.lower()
    for topic, pattern in TOPIC_PATTERNS.items():
        out[f"topic_{topic}"] = lower.str.contains(pattern, regex=True)
    topic_cols = [f"topic_{topic}" for topic in TOPIC_PATTERNS]
    out["topic"] = out[topic_cols].idxmax(axis=1).str.replace("topic_", "", regex=False)
    out.loc[~out[topic_cols].any(axis=1), "topic"] = "other"
    return out


def build_summary(metrics: dict[str, Any], errors: pd.DataFrame, high_confidence_errors: pd.DataFrame) -> str:
    matrix = metrics["confusion_matrix"]
    error_by_direction = errors["error_direction"].value_counts().to_dict()
    topic_counts = errors["topic"].value_counts().head(10)
    high_conf = high_confidence_errors.head(12)

    topic_lines = "\n".join(
        f"- {topic}: {count}"
        for topic, count in topic_counts.items()
    )
    high_conf_lines = "\n".join(
        f"- true={row.label}, pred={row.predicted_label}, margin={row.confidence_margin:.3f}: {row.headline}"
        for row in high_conf.itertuples()
    )
    if not high_conf_lines:
        high_conf_lines = "- None"

    return f"""# Error Analysis Summary

## OOF Metrics

- Accuracy: {metrics["accuracy"]:.4f}
- Macro F1: {metrics["macro_f1"]:.4f}
- ROC-AUC: {metrics["roc_auc"]:.4f}
- Threshold: {metrics["threshold"]:.6f}
- Total errors: {metrics["num_errors"]} / {metrics["num_examples"]}

## Confusion Matrix

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | {matrix[0][0]} | {matrix[0][1]} |
| NBC | {matrix[1][0]} | {matrix[1][1]} |

## Error Direction

- FoxNews predicted as NBC: {error_by_direction.get("FoxNews->NBC", 0)}
- NBC predicted as FoxNews: {error_by_direction.get("NBC->FoxNews", 0)}

## Error Topics

{topic_lines}

## Highest-Confidence Errors

{high_conf_lines}
"""


def run_error_analysis(data_path: Path) -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    work, dataset_summary = load_raw_clean_dataset(data_path)
    labels = work["label"].tolist()
    raw_texts = work["headline"].tolist()
    clean_texts = work["clean_headline"].tolist()

    labels_array = np.asarray(labels)
    oof_scores = np.zeros(len(work), dtype=float)
    fold_ids = np.zeros(len(work), dtype=int)
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

    threshold, _ = tune_threshold(oof_scores, labels)
    preds = predict_with_threshold(oof_scores, threshold)
    binary_targets = np.asarray([1 if label == LABELS[1] else 0 for label in labels], dtype=int)

    predictions = work.copy()
    predictions["fold"] = fold_ids
    predictions["score_nbc"] = oof_scores
    predictions["predicted_label"] = preds
    predictions["correct"] = predictions["label"] == predictions["predicted_label"]
    predictions["error_direction"] = predictions["label"] + "->" + predictions["predicted_label"]
    predictions.loc[predictions["correct"], "error_direction"] = ""
    predictions = add_analysis_columns(predictions, threshold)

    errors = predictions[~predictions["correct"]].copy()
    errors = errors.sort_values("confidence_margin", ascending=False)
    high_confidence_errors = errors.head(100).copy()

    metrics = {
        "dataset_path": str(data_path),
        "dataset": dataset_summary,
        "model": "headline_simple_meta_ensemble_recreated_oof",
        "classifier_c_values": list(ENSEMBLE_C_VALUES),
        "feature_config": {
            "word_max_features": WORD_MAX_FEATURES,
            "char_max_features": CHAR_MAX_FEATURES,
            "meta_features": int(headline_meta_features(["example headline"]).shape[1]),
            "word_ngram_range": list(WORD_NGRAM_RANGE),
            "char_ngram_range": list(CHAR_NGRAM_RANGE),
            "text_mode": "headline",
        },
        "threshold": float(threshold),
        "num_examples": int(len(predictions)),
        "num_errors": int(len(errors)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=LABELS, average="macro")),
        "roc_auc": float(roc_auc_score(binary_targets, oof_scores)),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(labels, preds, labels=LABELS).tolist(),
        "error_direction_counts": errors["error_direction"].value_counts().to_dict(),
        "error_topic_counts": errors["topic"].value_counts().to_dict(),
    }

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    errors.to_csv(ERRORS_PATH, index=False)
    high_confidence_errors.to_csv(HIGH_CONFIDENCE_ERRORS_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(build_summary(metrics, errors, high_confidence_errors), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    metrics = run_error_analysis(args.data)
    print(json.dumps(metrics, indent=2))
    print(f"saved predictions to {PREDICTIONS_PATH}")
    print(f"saved errors to {ERRORS_PATH}")
    print(f"saved high-confidence errors to {HIGH_CONFIDENCE_ERRORS_PATH}")
    print(f"saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
