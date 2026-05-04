from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass, replace
import html
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any
from urllib.parse import urlparse
import zlib

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "url_with_headlines.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
WEIGHTS_PATH = ARTIFACTS_DIR / "model.pt"
METRICS_PATH = ARTIFACTS_DIR / "model_search_metrics.json"
SUMMARY_PATH = ARTIFACTS_DIR / "model_improvement_summary.md"
MODEL_PATH = ROOT / "submission" / "model.py"
PREPROCESS_PATH = ROOT / "submission" / "preprocess.py"

LABELS = ["FoxNews", "NBC"]
RANDOM_STATE = 42
WORD_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
HEADLINE_COLUMNS = [
    "headline",
    "scraped_headline",
    "alternative_headline",
    "title",
]


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    word_max_features: int
    char_max_features: int
    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]
    min_df: int = 1
    sublinear_tf: bool = True
    text_mode: str = "headline"


@dataclass(frozen=True)
class ClassifierConfig:
    kind: str
    c: float
    class_weight: str | None = "balanced"


@dataclass(frozen=True)
class SearchRun:
    feature: FeatureConfig
    classifier: ClassifierConfig
    summary: dict[str, Any]


def clean_text(text: str) -> str:
    text = html.unescape(str(text)).strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(
        r"\s+[-|:]\s+"
        r"(fox news|fox business|nbc news|nbc news now|nbc select|msnbc|today)"
        r"\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


def infer_label(url: str) -> str:
    url = str(url).lower()
    if "foxnews.com" in url:
        return "FoxNews"
    if "nbcnews.com" in url:
        return "NBC"
    raise ValueError(f"Unable to infer label from url: {url}")


def select_headlines(df: pd.DataFrame, labels: list[str] | None = None) -> list[str]:
    available = [column for column in HEADLINE_COLUMNS if column in df.columns]
    if not available:
        raise ValueError(f"Expected one of headline columns: {HEADLINE_COLUMNS}")

    headlines: list[str] = []
    for _, row in df[available].fillna("").astype(str).iterrows():
        pieces: list[str] = []
        seen: set[str] = set()
        for column in HEADLINE_COLUMNS:
            if column not in available:
                continue
            candidate = str(row[column]).strip()
            cleaned = clean_text(candidate)
            if cleaned and cleaned not in seen:
                pieces.append(candidate)
                seen.add(cleaned)
        headlines.append(" ".join(pieces))
    return headlines


def url_to_text(url: str, text_mode: str) -> str:
    if text_mode == "headline":
        return ""

    parsed = urlparse(str(url))
    path = parsed.path.lower()
    if text_mode == "headline_url_slug":
        path = path.rstrip("/").split("/")[-1]
    elif text_mode != "headline_url_path":
        raise ValueError(f"Unknown text_mode: {text_mode}")

    path = re.sub(r"\.(html?|print)$", " ", path)
    path = re.sub(r"[^a-z0-9]+", " ", path)
    path = re.sub(r"\b(foxnews|fox|nbcnews|nbc|www|com|print)\b", " ", path)
    return re.sub(r"\s+", " ", path).strip()


def build_model_text(headline: str, url: str = "", text_mode: str = "headline") -> str:
    headline_text = clean_text(headline)
    url_text = url_to_text(url, text_mode)
    if not url_text:
        return headline_text
    return f"{headline_text} {url_text}".strip()


def load_dataset(
    path: Path,
    drop_blank: bool = True,
    drop_duplicate_headlines: bool = True,
    text_mode: str = "headline",
) -> tuple[list[str], list[str], dict[str, Any]]:
    df = pd.read_csv(path)

    if "url" in df.columns:
        labels = [infer_label(url) for url in df["url"].tolist()]
    elif "label" in df.columns:
        labels = df["label"].astype(str).tolist()
    elif "source" in df.columns:
        labels = df["source"].astype(str).tolist()
    else:
        raise ValueError("Expected one of: url, label, source columns.")

    raw_headlines = select_headlines(df, labels)
    urls = df["url"].fillna("").astype(str).tolist() if "url" in df.columns else [""] * len(df)

    work = df.copy()
    work["_label"] = labels
    work["_clean_headline"] = [
        build_model_text(headline, url, text_mode=text_mode)
        for headline, url in zip(raw_headlines, urls)
    ]

    input_examples = len(work)
    blank_examples = int((work["_clean_headline"] == "").sum())

    if drop_blank:
        work = work[work["_clean_headline"] != ""].copy()

    conflicting = (
        work.groupby("_clean_headline")["_label"]
        .nunique()
        .loc[lambda series: series > 1]
        .index
    )
    conflict_examples = int(work["_clean_headline"].isin(conflicting).sum())
    if len(conflicting) > 0:
        work = work[~work["_clean_headline"].isin(conflicting)].copy()

    duplicate_examples = int(work.duplicated("_clean_headline").sum())
    if drop_duplicate_headlines:
        work = work.drop_duplicates("_clean_headline", keep="first").copy()

    texts = work["_clean_headline"].tolist()
    labels = work["_label"].tolist()

    summary = {
        "input_examples": input_examples,
        "used_examples": len(work),
        "blank_examples": blank_examples,
        "dropped_blank_examples": blank_examples if drop_blank else 0,
        "conflicting_duplicate_examples": conflict_examples,
        "duplicate_headline_examples": duplicate_examples,
        "dropped_duplicate_headline_examples": duplicate_examples
        if drop_duplicate_headlines
        else 0,
        "label_counts": {label: labels.count(label) for label in LABELS},
        "text_mode": text_mode,
    }
    return texts, labels, summary


def default_feature_configs(quick: bool) -> list[FeatureConfig]:
    configs = [
        FeatureConfig("current_sublinear", 10000, 15000, (1, 2), (3, 5)),
        FeatureConfig("larger_vocab", 15000, 25000, (1, 2), (3, 5)),
        FeatureConfig("word_trigrams", 20000, 30000, (1, 3), (3, 5)),
        FeatureConfig("char_3_6", 15000, 30000, (1, 2), (3, 6)),
        FeatureConfig("char_3_6_min_df_2", 15000, 25000, (1, 2), (3, 6), min_df=2),
    ]
    return configs[:2] if quick else configs


def default_classifier_configs(quick: bool) -> list[ClassifierConfig]:
    if quick:
        return [
            ClassifierConfig("logreg", 1.0),
            ClassifierConfig("logreg", 2.0),
            ClassifierConfig("linearsvc", 0.5),
            ClassifierConfig("linearsvc", 1.0),
        ]

    return [
        *(ClassifierConfig("logreg", c) for c in [0.5, 1.0, 2.0, 4.0, 8.0]),
        *(ClassifierConfig("linearsvc", c) for c in [0.25, 0.5, 1.0, 2.0]),
    ]


def expanded_feature_configs() -> list[FeatureConfig]:
    return [
        FeatureConfig("expanded_w30_c50_word13_char35", 30000, 50000, (1, 3), (3, 5)),
        FeatureConfig("expanded_w50_c80_word13_char35", 50000, 80000, (1, 3), (3, 5)),
        FeatureConfig("expanded_w30_c50_word14_char35", 30000, 50000, (1, 4), (3, 5)),
        FeatureConfig("expanded_w30_c50_word13_char25", 30000, 50000, (1, 3), (2, 5)),
        FeatureConfig("expanded_w30_c80_word13_char36", 30000, 80000, (1, 3), (3, 6)),
        FeatureConfig("expanded_w50_c80_word14_char36", 50000, 80000, (1, 4), (3, 6)),
        FeatureConfig("expanded_w30_c50_word13_char35_min_df_2", 30000, 50000, (1, 3), (3, 5), min_df=2),
        FeatureConfig("expanded_w50_c80_word13_char36_min_df_2", 50000, 80000, (1, 3), (3, 6), min_df=2),
    ]


def expanded_classifier_configs(c_values: list[float]) -> list[ClassifierConfig]:
    return [ClassifierConfig("linearsvc", c) for c in c_values]


def with_text_mode(configs: list[FeatureConfig], text_mode: str) -> list[FeatureConfig]:
    return [replace(config, text_mode=text_mode) for config in configs]


def make_word_vectorizer(config: FeatureConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=config.word_ngram_range,
        max_features=config.word_max_features,
        min_df=config.min_df,
        sublinear_tf=config.sublinear_tf,
    )


def make_char_vectorizer(config: FeatureConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        analyzer="char_wb",
        ngram_range=config.char_ngram_range,
        max_features=config.char_max_features,
        min_df=config.min_df,
        sublinear_tf=config.sublinear_tf,
    )


def fit_vectorizers(texts: list[str], config: FeatureConfig) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word_vectorizer = make_word_vectorizer(config)
    char_vectorizer = make_char_vectorizer(config)
    word_vectorizer.fit(texts)
    char_vectorizer.fit(texts)
    return word_vectorizer, char_vectorizer


def transform_texts(
    texts: list[str],
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
):
    return hstack(
        [
            word_vectorizer.transform(texts),
            char_vectorizer.transform(texts),
        ],
        format="csr",
    )


def make_classifier(config: ClassifierConfig):
    if config.kind == "logreg":
        return LogisticRegression(
            C=config.c,
            class_weight=config.class_weight,
            max_iter=5000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
    if config.kind == "linearsvc":
        return LinearSVC(
            C=config.c,
            class_weight=config.class_weight,
            max_iter=10000,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown classifier kind: {config.kind}")


def scores_for_positive_class(clf: Any, X) -> np.ndarray:
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    elif hasattr(clf, "predict_proba"):
        classes = [str(item) for item in clf.classes_]
        return np.asarray(clf.predict_proba(X)[:, classes.index(LABELS[1])], dtype=float)
    else:
        raise TypeError("Classifier must expose decision_function or predict_proba.")

    scores = np.asarray(scores, dtype=float)
    classes = [str(item) for item in clf.classes_]
    if scores.ndim == 2:
        return scores[:, classes.index(LABELS[1])]

    if len(classes) != 2:
        raise ValueError(f"Expected two classes, got: {classes}")
    return scores if classes[1] == LABELS[1] else -scores


def predict_with_threshold(scores: np.ndarray, threshold: float) -> list[str]:
    return np.where(scores >= threshold, LABELS[1], LABELS[0]).tolist()


def binary_targets(labels: list[str]) -> np.ndarray:
    return np.asarray([1 if label == LABELS[1] else 0 for label in labels], dtype=int)


def safe_roc_auc(labels: list[str], scores: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(binary_targets(labels), scores))
    except ValueError:
        return None


def label_metrics(labels: list[str], preds: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=LABELS, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, labels=LABELS, average="weighted")),
    }


def tune_threshold(scores: np.ndarray, labels: list[str]) -> tuple[float, float]:
    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return 0.0, 0.0

    candidates = [0.0, float(unique_scores[0] - 1.0), float(unique_scores[-1] + 1.0)]
    if unique_scores.size > 1:
        candidates.extend(((unique_scores[:-1] + unique_scores[1:]) / 2.0).tolist())

    best_threshold = 0.0
    best_accuracy = -1.0
    for threshold in candidates:
        preds = predict_with_threshold(scores, threshold)
        accuracy = accuracy_score(labels, preds)
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def evaluate_config(
    texts: list[str],
    labels: list[str],
    feature_config: FeatureConfig,
    classifier_config: ClassifierConfig,
    n_splits: int,
) -> SearchRun:
    labels_array = np.asarray(labels)
    oof_scores = np.zeros(len(labels), dtype=float)
    fold_default_accuracies: list[float] = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for train_idx, val_idx in skf.split(texts, labels):
        train_text = [texts[index] for index in train_idx]
        val_text = [texts[index] for index in val_idx]
        y_train = labels_array[train_idx].tolist()
        y_val = labels_array[val_idx].tolist()

        word_vectorizer, char_vectorizer = fit_vectorizers(train_text, feature_config)
        X_train = transform_texts(train_text, word_vectorizer, char_vectorizer)
        X_val = transform_texts(val_text, word_vectorizer, char_vectorizer)

        clf = make_classifier(classifier_config)
        clf.fit(X_train, y_train)

        fold_scores = scores_for_positive_class(clf, X_val)
        oof_scores[val_idx] = fold_scores
        fold_default_accuracies.append(
            float(accuracy_score(y_val, predict_with_threshold(fold_scores, 0.0)))
        )

    threshold, threshold_accuracy = tune_threshold(oof_scores, labels)
    default_preds = predict_with_threshold(oof_scores, 0.0)
    threshold_preds = predict_with_threshold(oof_scores, threshold)
    default_metrics = label_metrics(labels, default_preds)
    threshold_metrics = label_metrics(labels, threshold_preds)

    summary = {
        "feature_config": config_to_dict(feature_config),
        "classifier_config": config_to_dict(classifier_config),
        "fold_default_accuracy_mean": float(np.mean(fold_default_accuracies)),
        "fold_default_accuracy_std": float(np.std(fold_default_accuracies)),
        "oof_default_accuracy": default_metrics["accuracy"],
        "oof_threshold_accuracy": float(threshold_accuracy),
        "oof_default_macro_f1": default_metrics["macro_f1"],
        "oof_default_weighted_f1": default_metrics["weighted_f1"],
        "oof_threshold_macro_f1": threshold_metrics["macro_f1"],
        "oof_threshold_weighted_f1": threshold_metrics["weighted_f1"],
        "oof_roc_auc": safe_roc_auc(labels, oof_scores),
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
    return SearchRun(feature_config, classifier_config, summary)


def evaluate_config_on_holdout(
    train_texts: list[str],
    train_labels: list[str],
    validation_texts: list[str],
    validation_labels: list[str],
    feature_config: FeatureConfig,
    classifier_config: ClassifierConfig,
) -> SearchRun:
    word_vectorizer, char_vectorizer = fit_vectorizers(train_texts, feature_config)
    X_train = transform_texts(train_texts, word_vectorizer, char_vectorizer)
    X_validation = transform_texts(validation_texts, word_vectorizer, char_vectorizer)

    clf = make_classifier(classifier_config)
    clf.fit(X_train, train_labels)

    scores = scores_for_positive_class(clf, X_validation)
    threshold, threshold_accuracy = tune_threshold(scores, validation_labels)
    default_preds = predict_with_threshold(scores, 0.0)
    threshold_preds = predict_with_threshold(scores, threshold)
    default_metrics = label_metrics(validation_labels, default_preds)
    threshold_metrics = label_metrics(validation_labels, threshold_preds)

    summary = {
        "validation_protocol": "holdout_csv",
        "feature_config": config_to_dict(feature_config),
        "classifier_config": config_to_dict(classifier_config),
        "fold_default_accuracy_mean": default_metrics["accuracy"],
        "fold_default_accuracy_std": 0.0,
        "oof_default_accuracy": default_metrics["accuracy"],
        "oof_threshold_accuracy": float(threshold_accuracy),
        "oof_default_macro_f1": default_metrics["macro_f1"],
        "oof_default_weighted_f1": default_metrics["weighted_f1"],
        "oof_threshold_macro_f1": threshold_metrics["macro_f1"],
        "oof_threshold_weighted_f1": threshold_metrics["weighted_f1"],
        "oof_roc_auc": safe_roc_auc(validation_labels, scores),
        "threshold": float(threshold),
        "classification_report": classification_report(
            validation_labels,
            threshold_preds,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(
            validation_labels,
            threshold_preds,
            labels=LABELS,
        ).tolist(),
    }
    return SearchRun(feature_config, classifier_config, summary)


def config_to_dict(config: FeatureConfig | ClassifierConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, tuple):
            data[key] = list(value)
    return data


def terms_by_index(vectorizer: TfidfVectorizer) -> list[str]:
    terms: list[str | None] = [None] * len(vectorizer.vocabulary_)
    for term, index in vectorizer.vocabulary_.items():
        terms[index] = term
    if any(term is None for term in terms):
        raise RuntimeError("Vectorizer vocabulary has missing feature indices.")
    return [str(term) for term in terms]


def positive_coef_intercept(clf: Any) -> tuple[np.ndarray, float]:
    classes = [str(item) for item in clf.classes_]
    coef = np.asarray(clf.coef_, dtype=float)
    intercept = np.asarray(clf.intercept_, dtype=float)

    if coef.shape[0] == 1:
        sign = 1.0 if classes[1] == LABELS[1] else -1.0
        return sign * coef[0], float(sign * intercept[0])

    positive_index = classes.index(LABELS[1])
    negative_index = classes.index(LABELS[0])
    return (
        coef[positive_index] - coef[negative_index],
        float(intercept[positive_index] - intercept[negative_index]),
    )


def export_state_dict(clf: Any, num_features: int, threshold: float) -> dict[str, Any]:
    import torch

    coef, intercept = positive_coef_intercept(clf)
    weights = torch.zeros((len(LABELS), num_features), dtype=torch.float32)
    bias = torch.zeros(len(LABELS), dtype=torch.float32)

    positive_index = LABELS.index(LABELS[1])
    weights[positive_index] = torch.tensor(coef, dtype=torch.float32)
    bias[positive_index] = torch.tensor(intercept - threshold, dtype=torch.float32)

    return {
        "linear.weight": weights,
        "linear.bias": bias,
    }


def encode_payload(payload: dict[str, Any]) -> str:
    return base64.b64encode(
        zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), level=9)
    ).decode("ascii")


def build_model_py(num_features: int) -> str:
    return f'''from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn

NUM_FEATURES = {num_features}
NUM_CLASSES = 2
LABELS = ["FoxNews", "NBC"]


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(NUM_FEATURES, NUM_CLASSES)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.linear(batch.float())

    def predict(self, batch: Iterable[torch.Tensor]) -> List[str]:
        if isinstance(batch, torch.Tensor):
            features = batch.float()
        else:
            features = torch.stack([item.float() for item in batch], dim=0)
        with torch.no_grad():
            logits = self.forward(features)
            preds = torch.argmax(logits, dim=-1).tolist()
        return [LABELS[index] for index in preds]


def get_model() -> Model:
    return Model()
'''


def build_preprocess_py(
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    feature_config: FeatureConfig,
) -> str:
    word_terms = terms_by_index(word_vectorizer)
    char_terms = terms_by_index(char_vectorizer)
    payload = {
        "word_terms": word_terms,
        "word_idf": [float(value) for value in word_vectorizer.idf_],
        "char_terms": char_terms,
        "char_idf": [float(value) for value in char_vectorizer.idf_],
    }
    encoded = encode_payload(payload)
    wrapped = "\n".join(f'    "{chunk}"' for chunk in textwrap.wrap(encoded, width=88))
    num_features = len(word_terms) + len(char_terms)
    word_min, word_max = feature_config.word_ngram_range
    char_min, char_max = feature_config.char_ngram_range
    url_import = "" if feature_config.text_mode == "headline" else "from urllib.parse import urlparse\n"
    if feature_config.text_mode == "headline":
        url_helpers = '''def build_model_text(headline: str, url: str = "") -> str:
    return clean_text(headline)
'''
    else:
        url_helpers = '''def url_to_text(url: str) -> str:
    parsed = urlparse(str(url))
    path = parsed.path.lower()
    if TEXT_MODE == "headline_url_slug":
        path = path.rstrip("/").split("/")[-1]
    elif TEXT_MODE != "headline_url_path":
        raise ValueError(f"Unknown TEXT_MODE: {TEXT_MODE}")

    path = re.sub(r"\\.(html?|print)$", " ", path)
    path = re.sub(r"[^a-z0-9]+", " ", path)
    path = re.sub(r"\\b(foxnews|fox|nbcnews|nbc|www|com|print)\\b", " ", path)
    return re.sub(r"\\s+", " ", path).strip()


def build_model_text(headline: str, url: str = "") -> str:
    headline_text = clean_text(headline)
    url_text = url_to_text(url)
    if not url_text:
        return headline_text
    return f"{headline_text} {url_text}".strip()
'''

    return f'''from __future__ import annotations

import base64
import html
import json
import math
import re
import zlib
{url_import.rstrip()}
from typing import Dict, List, Tuple

import pandas as pd
import torch

NUM_WORD_FEATURES = {len(word_terms)}
NUM_CHAR_FEATURES = {len(char_terms)}
NUM_FEATURES = {num_features}
WORD_TOKEN_PATTERN = re.compile(r"(?u)\\b\\w\\w+\\b")
WORD_NGRAM_RANGE = ({word_min}, {word_max})
CHAR_NGRAM_RANGE = ({char_min}, {char_max})
SUBLINEAR_TF = {feature_config.sublinear_tf}
HEADLINE_COLUMNS = ["headline", "scraped_headline", "alternative_headline", "title"]
TEXT_MODE = "{feature_config.text_mode}"

_VECTORIZER_B64 = (
{wrapped}
)
_WORD_VOCAB: Dict[str, int] | None = None
_WORD_IDF: List[float] | None = None
_CHAR_VOCAB: Dict[str, int] | None = None
_CHAR_IDF: List[float] | None = None


def _load_vectorizers() -> Tuple[Dict[str, int], List[float], Dict[str, int], List[float]]:
    global _WORD_VOCAB, _WORD_IDF, _CHAR_VOCAB, _CHAR_IDF
    if _WORD_VOCAB is None or _WORD_IDF is None or _CHAR_VOCAB is None or _CHAR_IDF is None:
        payload = json.loads(zlib.decompress(base64.b64decode(_VECTORIZER_B64)).decode("utf-8"))
        _WORD_VOCAB = {{term: index for index, term in enumerate(payload["word_terms"])}}
        _WORD_IDF = [float(value) for value in payload["word_idf"]]
        _CHAR_VOCAB = {{term: index for index, term in enumerate(payload["char_terms"])}}
        _CHAR_IDF = [float(value) for value in payload["char_idf"]]
    return _WORD_VOCAB, _WORD_IDF, _CHAR_VOCAB, _CHAR_IDF


def clean_text(text: str) -> str:
    text = html.unescape(str(text)).strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\u00a0", " ")
    text = text.replace("\\u2019", "'").replace("\\u2018", "'")
    text = text.replace("\\u201c", '"').replace("\\u201d", '"')
    text = text.replace("\\u2013", "-").replace("\\u2014", "-")
    text = re.sub(
        r"\\s+[-|:]\\s+"
        r"(fox news|fox business|nbc news|nbc news now|nbc select|msnbc|today)"
        r"\\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\\s+", " ", text).strip()


{url_helpers.rstrip()}


def _tf(count: float) -> float:
    return 1.0 + math.log(count) if SUBLINEAR_TF else count


def tokenize_words(text: str) -> List[str]:
    return WORD_TOKEN_PATTERN.findall(clean_text(text))


def iter_word_ngrams(tokens: List[str]):
    min_n, max_n = WORD_NGRAM_RANGE
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i : i + n])


def iter_char_wb_ngrams(text: str):
    min_n, max_n = CHAR_NGRAM_RANGE
    normalized = re.sub(r"\\s+", " ", clean_text(text))
    for word in normalized.split():
        padded = f" {{word}} "
        length = len(padded)
        for n in range(min_n, max_n + 1):
            for i in range(length - n + 1):
                yield padded[i : i + n]


def infer_label(url: str) -> str:
    url = str(url).lower()
    if "foxnews.com" in url:
        return "FoxNews"
    if "nbcnews.com" in url:
        return "NBC"
    raise ValueError(f"Unable to infer label from url: {{url}}")


def select_headlines(df: pd.DataFrame, labels: List[str] | None = None) -> List[str]:
    available = [column for column in HEADLINE_COLUMNS if column in df.columns]
    if not available:
        raise ValueError(f"Expected one of headline columns: {{HEADLINE_COLUMNS}}")

    headlines: List[str] = []
    for _, row in df[available].fillna("").astype(str).iterrows():
        pieces: List[str] = []
        seen: set[str] = set()
        for column in HEADLINE_COLUMNS:
            if column not in available:
                continue
            candidate = str(row[column]).strip()
            cleaned = clean_text(candidate)
            if cleaned and cleaned not in seen:
                pieces.append(candidate)
                seen.add(cleaned)
        headlines.append(" ".join(pieces))
    return headlines


def _normalize(features: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(features)
    if norm > 0:
        features /= norm
    return features


def vectorize_headline(headline: str) -> torch.Tensor:
    word_vocab, word_idf, char_vocab, char_idf = _load_vectorizers()
    word_counts: Dict[int, float] = {{}}
    char_counts: Dict[int, float] = {{}}

    for term in iter_word_ngrams(tokenize_words(headline)):
        index = word_vocab.get(term)
        if index is not None:
            word_counts[index] = word_counts.get(index, 0.0) + 1.0

    for term in iter_char_wb_ngrams(headline):
        index = char_vocab.get(term)
        if index is not None:
            char_counts[index] = char_counts.get(index, 0.0) + 1.0

    word_features = torch.zeros(NUM_WORD_FEATURES, dtype=torch.float32)
    for index, count in word_counts.items():
        word_features[index] = _tf(count) * word_idf[index]
    _normalize(word_features)

    char_features = torch.zeros(NUM_CHAR_FEATURES, dtype=torch.float32)
    for index, count in char_counts.items():
        char_features[index] = _tf(count) * char_idf[index]
    _normalize(char_features)

    return torch.cat([word_features, char_features])


def prepare_data(path: str) -> Tuple[List[torch.Tensor], List[str]]:
    df = pd.read_csv(path)

    if "url" in df.columns:
        labels = [infer_label(url) for url in df["url"].tolist()]
    elif "label" in df.columns:
        labels = df["label"].astype(str).tolist()
    elif "source" in df.columns:
        labels = df["source"].astype(str).tolist()
    else:
        raise ValueError("Expected one of: url, label, source columns.")

    headlines = select_headlines(df, labels)
    urls = df["url"].fillna("").astype(str).tolist() if "url" in df.columns else [""] * len(df)
    X = [vectorize_headline(build_model_text(text, url)) for text, url in zip(headlines, urls)]
    return X, labels
'''


def fit_final_and_export(
    texts: list[str],
    labels: list[str],
    best_run: SearchRun,
    export: bool,
) -> dict[str, Any]:
    word_vectorizer, char_vectorizer = fit_vectorizers(texts, best_run.feature)
    X_all = transform_texts(texts, word_vectorizer, char_vectorizer)
    clf = make_classifier(best_run.classifier)
    clf.fit(X_all, labels)

    num_features = int(X_all.shape[1])
    threshold = float(best_run.summary["threshold"])
    if export:
        import torch

        torch.save(export_state_dict(clf, num_features, threshold), WEIGHTS_PATH)
        MODEL_PATH.write_text(build_model_py(num_features), encoding="utf-8")
        PREPROCESS_PATH.write_text(
            build_preprocess_py(word_vectorizer, char_vectorizer, best_run.feature),
            encoding="utf-8",
        )

    return {
        "num_features": num_features,
        "threshold_exported_as_bias_shift": threshold,
        "weights_path": str(WEIGHTS_PATH),
        "model_path": str(MODEL_PATH),
        "preprocess_path": str(PREPROCESS_PATH),
    }


def build_improvement_summary(metrics: dict[str, Any]) -> str:
    best = metrics["best_model"]
    feature = best["feature_config"]
    classifier = best["classifier_config"]
    report = best["classification_report"]
    matrix = best["confusion_matrix"]
    dataset = metrics["dataset"]
    validation_dataset = metrics.get("validation_dataset")
    validation_protocol = metrics.get("validation_protocol", "stratified_cv")

    roc_auc = best.get("oof_roc_auc")
    roc_auc_text = "not available" if roc_auc is None else f"{roc_auc:.4f}"
    if validation_protocol == "holdout_csv":
        protocol_text = (
            "holdout accuracy on the separate validation CSV, using the training CSV only for fitting"
        )
        result_heading = "Holdout Validation Results"
        threshold_text = "Tuned the decision threshold on held-out validation scores to improve final accuracy."
        validation_text = (
            f"\n- Validation dataset used after cleaning: {validation_dataset['used_examples']} examples"
            if validation_dataset
            else ""
        )
    else:
        protocol_text = "5-fold cross-validated accuracy, matching the course leaderboard metric"
        result_heading = "Cross-Validated Results"
        threshold_text = "Tuned the decision threshold on out-of-fold scores to improve final accuracy."
        validation_text = ""

    return f"""# News Source Classification Model Improvement Summary

## Final Selected Model

- Model: {classifier["kind"]} with C={classifier["c"]} and class_weight={classifier["class_weight"]}
- Features: TF-IDF word {feature["word_ngram_range"]} n-grams plus character {feature["char_ngram_range"]} n-grams
- Vocabulary: {feature["word_max_features"]} word features + {feature["char_max_features"]} character features
- TF scaling: sublinear_tf={feature["sublinear_tf"]}
- Text mode: {feature.get("text_mode", "headline")}
- Threshold: {best["threshold"]:.6f}

## Evaluation Protocol

- Selection metric: {protocol_text}
- Diagnostic metrics: macro F1, weighted F1, per-class precision/recall/F1, confusion matrix, ROC-AUC
- Training dataset used after cleaning: {dataset["used_examples"]} examples{validation_text}
- Dropped examples: {dataset["dropped_blank_examples"]} blank headlines and {dataset["dropped_duplicate_headline_examples"]} duplicate headlines

## {result_heading}

- Accuracy: {best["oof_threshold_accuracy"]:.4f}
- Macro F1: {best["oof_threshold_macro_f1"]:.4f}
- Weighted F1: {best["oof_threshold_weighted_f1"]:.4f}
- ROC-AUC: {roc_auc_text}
- FoxNews precision/recall/F1: {report["FoxNews"]["precision"]:.4f} / {report["FoxNews"]["recall"]:.4f} / {report["FoxNews"]["f1-score"]:.4f}
- NBC precision/recall/F1: {report["NBC"]["precision"]:.4f} / {report["NBC"]["recall"]:.4f} / {report["NBC"]["f1-score"]:.4f}

## Confusion Matrix

Rows are true labels and columns are predicted labels.

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | {matrix[0][0]} | {matrix[0][1]} |
| NBC | {matrix[1][0]} | {matrix[1][1]} |

## Main Improvements

- Replaced the fixed single-split experiment with 5-fold stratified cross-validation.
- Switched the best classifier from Logistic Regression to LinearSVC after empirical comparison.
- Increased TF-IDF capacity and added word trigrams while keeping character n-grams.
- Added optional URL-slug text features with source domains stripped.
- Applied the same text cleaning during training and submission preprocessing.
- Removed blank and duplicate cleaned headlines from training.
- {threshold_text}
- Added macro F1, weighted F1, ROC-AUC, and confusion-matrix diagnostics for reporting while keeping accuracy as the final selection target.
"""


def run_search(
    texts: list[str],
    labels: list[str],
    folds: int,
    quick: bool,
    feature_configs: list[FeatureConfig] | None = None,
    classifier_configs: list[ClassifierConfig] | None = None,
    validation_texts: list[str] | None = None,
    validation_labels: list[str] | None = None,
) -> list[SearchRun]:
    runs: list[SearchRun] = []
    if feature_configs is None:
        feature_configs = default_feature_configs(quick)
    if classifier_configs is None:
        classifier_configs = default_classifier_configs(quick)
    total = len(feature_configs) * len(classifier_configs)

    for run_index, feature_config in enumerate(feature_configs, start=1):
        for classifier_config in classifier_configs:
            display_index = len(runs) + 1
            print(
                f"[{display_index}/{total}] {feature_config.name} "
                f"+ {classifier_config.kind}(C={classifier_config.c})"
            )
            if validation_texts is not None and validation_labels is not None:
                run = evaluate_config_on_holdout(
                    texts,
                    labels,
                    validation_texts,
                    validation_labels,
                    feature_config,
                    classifier_config,
                )
            else:
                run = evaluate_config(texts, labels, feature_config, classifier_config, folds)
            runs.append(run)
            print(
                "    "
                f"oof_acc={run.summary['oof_default_accuracy']:.4f} "
                f"threshold_acc={run.summary['oof_threshold_accuracy']:.4f} "
                f"threshold={run.summary['threshold']:.4f}"
            )

    runs.sort(
        key=lambda run: (
            run.summary["oof_threshold_accuracy"],
            run.summary["oof_default_accuracy"],
        ),
        reverse=True,
    )
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validate TF-IDF headline classifiers and export the best submission model."
    )
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=None,
        help="Optional pseudo-hidden validation CSV. If set, train on --csv and select models on this file.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Run a smaller search space.")
    parser.add_argument(
        "--expanded-tfidf",
        action="store_true",
        help="Search larger TF-IDF vocabularies and denser LinearSVC C values.",
    )
    parser.add_argument("--no-export", action="store_true", help="Do not rewrite submission artifacts.")
    parser.add_argument(
        "--refine-word-trigrams-svc",
        action="store_true",
        help="Only search LinearSVC C values with the best word-trigram TF-IDF feature setup.",
    )
    parser.add_argument(
        "--svc-c-values",
        type=float,
        nargs="+",
        default=[0.35, 0.4, 0.45, 0.5, 0.6, 0.75],
        help="C values for --refine-word-trigrams-svc.",
    )
    parser.add_argument(
        "--expanded-c-values",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5],
        help="LinearSVC C values for --expanded-tfidf.",
    )
    parser.add_argument(
        "--text-mode",
        choices=["headline", "headline_url_slug", "headline_url_path"],
        default="headline",
        help="Text used as model input. URL modes strip the source domain and append URL slug/path tokens.",
    )
    parser.add_argument("--keep-blank", action="store_true", help="Keep rows with blank headlines.")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate cleaned headlines.",
    )
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.refine_word_trigrams_svc and args.expanded_tfidf:
        raise ValueError("Choose either --refine-word-trigrams-svc or --expanded-tfidf, not both.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    texts, labels, dataset_summary = load_dataset(
        args.csv,
        drop_blank=not args.keep_blank,
        drop_duplicate_headlines=not args.keep_duplicates,
        text_mode=args.text_mode,
    )
    print(json.dumps({"dataset": dataset_summary}, indent=2))

    validation_texts = None
    validation_labels = None
    validation_dataset_summary = None
    if args.validation_csv is not None:
        validation_texts, validation_labels, validation_dataset_summary = load_dataset(
            args.validation_csv,
            drop_blank=not args.keep_blank,
            drop_duplicate_headlines=not args.keep_duplicates,
            text_mode=args.text_mode,
        )
        print(json.dumps({"validation_dataset": validation_dataset_summary}, indent=2))

    feature_configs = None
    classifier_configs = None
    search_mode = "quick" if args.quick else "full"
    if args.refine_word_trigrams_svc:
        search_mode = "word_trigrams_linearsvc_refine"
        feature_configs = [
            FeatureConfig("word_trigrams", 20000, 30000, (1, 3), (3, 5)),
        ]
        classifier_configs = [
            ClassifierConfig("linearsvc", c) for c in args.svc_c_values
        ]
    elif args.expanded_tfidf:
        search_mode = "expanded_tfidf_linearsvc"
        feature_configs = expanded_feature_configs()
        classifier_configs = expanded_classifier_configs(args.expanded_c_values)
    if feature_configs is None:
        feature_configs = default_feature_configs(args.quick)
    if classifier_configs is None:
        classifier_configs = default_classifier_configs(args.quick)
    if feature_configs is not None:
        feature_configs = with_text_mode(feature_configs, args.text_mode)

    runs = run_search(
        texts,
        labels,
        folds=args.folds,
        quick=args.quick,
        feature_configs=feature_configs,
        classifier_configs=classifier_configs,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
    )
    best_run = runs[0]
    export_summary = fit_final_and_export(
        texts,
        labels,
        best_run,
        export=not args.no_export,
    )

    metrics = {
        "dataset_path": str(args.csv),
        "dataset": dataset_summary,
        "validation_dataset_path": str(args.validation_csv) if args.validation_csv else None,
        "validation_dataset": validation_dataset_summary,
        "validation_protocol": "holdout_csv" if args.validation_csv else "stratified_cv",
        "text_mode": args.text_mode,
        "cv_folds": args.folds,
        "quick": args.quick,
        "search_mode": search_mode,
        "selection_metric": "oof_threshold_accuracy",
        "diagnostic_metrics": [
            "macro_f1",
            "weighted_f1",
            "per_class_precision_recall_f1",
            "confusion_matrix",
            "roc_auc",
        ],
        "best_model": best_run.summary,
        "final_export": export_summary,
        "all_results": [run.summary for run in runs[: args.top_k]],
    }
    metrics_path = METRICS_PATH
    if args.no_export:
        metrics_path = ARTIFACTS_DIR / "model_search_preview_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary_path = SUMMARY_PATH
    if args.no_export:
        summary_path = ARTIFACTS_DIR / "model_search_preview_summary.md"
    summary_path.write_text(build_improvement_summary(metrics), encoding="utf-8")

    print("\nBest model")
    print(json.dumps(best_run.summary, indent=2))
    if args.no_export:
        print(f"\nsaved metrics to {metrics_path}; export skipped")
    else:
        print(f"\nsaved weights to {WEIGHTS_PATH}")
        print(f"rewrote model constants in {MODEL_PATH}")
        print(f"rewrote preprocess constants in {PREPROCESS_PATH}")
        print(f"saved metrics to {metrics_path}")
    print(f"saved improvement summary to {summary_path}")


if __name__ == "__main__":
    main()
