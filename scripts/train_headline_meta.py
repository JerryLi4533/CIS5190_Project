from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any
import zlib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch

from train_model_search import (
    ARTIFACTS_DIR,
    DATA_PATH,
    LABELS,
    MODEL_PATH,
    PREPROCESS_PATH,
    RANDOM_STATE,
    WEIGHTS_PATH,
    build_model_py,
    build_model_text,
    infer_label,
    positive_coef_intercept,
    predict_with_threshold,
    select_headlines,
    terms_by_index,
    tune_threshold,
)

METRICS_PATH = ARTIFACTS_DIR / "headline_meta_metrics.json"
SUMMARY_PATH = ARTIFACTS_DIR / "headline_meta_summary.md"

WORD_MAX_FEATURES = 30000
CHAR_MAX_FEATURES = 50000
WORD_NGRAM_RANGE = (1, 3)
CHAR_NGRAM_RANGE = (2, 5)
SVC_C = 0.8
FOLDS = 5


SOURCE_SUFFIX_PATTERN = re.compile(
    r"\s+[-|:]\s+"
    r"(fox news|fox business|nbc news|nbc news now|nbc select|msnbc|today)"
    r"\s*$",
    flags=re.IGNORECASE,
)

POLITICS_WORDS = {
    "trump",
    "biden",
    "white house",
    "congress",
    "senate",
    "house",
    "republican",
    "democrat",
    "gop",
    "election",
    "campaign",
    "border",
    "court",
    "judge",
    "supreme court",
    "lawmakers",
}
LOCATION_WORDS = {
    "america",
    "american",
    "u.s.",
    "us",
    "china",
    "russia",
    "ukraine",
    "israel",
    "gaza",
    "iran",
    "mexico",
    "new york",
    "california",
    "florida",
    "texas",
    "washington",
    "chicago",
    "los angeles",
}
SOFT_ATTRIBUTION_WORDS = {
    "reportedly",
    "apparently",
    "allegedly",
    "likely",
    "could",
    "may",
    "might",
}
APPROX_WORDS = {
    "about",
    "around",
    "nearly",
    "almost",
    "roughly",
    "approximately",
    "over",
    "under",
    "more than",
    "less than",
    "at least",
    "up to",
}
ACTION_VERBS = {
    "says",
    "said",
    "tells",
    "warns",
    "slams",
    "blasts",
    "reveals",
    "claims",
    "admits",
    "accuses",
    "calls",
    "urges",
}
OPINION_WORDS = {
    "shocking",
    "stunning",
    "outrage",
    "controversial",
    "exclusive",
    "bizarre",
    "wild",
    "dramatic",
    "heartbreaking",
    "surprising",
}
BE_FORMS = r"(is|are|was|were|be|been|being|gets?|got)"


def strip_source_suffix(text: str) -> str:
    text = html.unescape(str(text)).strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = SOURCE_SUFFIX_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text: str, terms: set[str]) -> float:
    return float(any(term in text for term in terms))


def count_any(text: str, terms: set[str]) -> float:
    return float(sum(text.count(term) for term in terms))


def headline_meta_features(texts: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for text in texts:
        raw = strip_source_suffix(text)
        lower = raw.lower()
        chars = len(raw)
        words = re.findall(r"\b\w+\b", raw)
        lower_words = re.findall(r"\b\w+\b", lower)
        letters = sum(1 for char in raw if char.isalpha())
        uppercase = sum(1 for char in raw if char.isupper())
        punctuation = sum(1 for char in raw if char in "!?;:'\".,-$%&()[]")
        titlecase_words = sum(1 for word in words if word[:1].isupper() and word[1:].islower())
        allcaps_words = sum(1 for word in words if len(word) > 1 and word.isupper())
        person_names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", raw)
        passive = re.search(rf"\b{BE_FORMS}\s+\w+(ed|en)\b", lower)
        quoted = '"' in raw or "'" in raw
        exact_number = bool(re.search(r"\b\d{2,}\b", raw))
        percent_or_money = "%" in raw or "$" in raw
        starts_with_name = bool(re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", raw))
        starts_with_quote = raw.startswith(('"', "'"))
        starts_with_number = bool(re.match(r"^\d", raw))
        word_count = len(words)
        rows.append(
            [
                math.log1p(chars),
                math.log1p(word_count),
                sum(len(word) for word in words) / max(word_count, 1),
                uppercase / max(letters, 1),
                titlecase_words / max(word_count, 1),
                allcaps_words / max(word_count, 1),
                punctuation / max(chars, 1),
                raw.count("?") / max(word_count, 1),
                raw.count("!") / max(word_count, 1),
                raw.count("'") / max(word_count, 1),
                raw.count('"') / max(word_count, 1),
                raw.count(":") / max(word_count, 1),
                raw.count(",") / max(word_count, 1),
                raw.count(".") / max(word_count, 1),
                raw.count("-") / max(word_count, 1),
                float(any(char.isdigit() for char in raw)),
                sum(char.isdigit() for char in raw) / max(chars, 1),
                float(exact_number),
                float(percent_or_money),
                count_any(lower, APPROX_WORDS),
                float(passive is not None),
                count_any(lower, ACTION_VERBS),
                float(quoted),
                float(starts_with_name),
                min(len(person_names), 3) / 3.0,
                float(starts_with_quote),
                float(starts_with_number),
                contains_any(lower, POLITICS_WORDS),
                count_any(lower, POLITICS_WORDS) / max(word_count, 1),
                contains_any(lower, LOCATION_WORDS),
                count_any(lower, LOCATION_WORDS) / max(word_count, 1),
                contains_any(lower, SOFT_ATTRIBUTION_WORDS),
                count_any(lower, OPINION_WORDS),
                float(any(word.endswith("ing") for word in lower_words)),
                float(any(word.endswith("ed") for word in lower_words)),
                float(any(word in {"best", "worst", "top", "most", "least", "more", "less"} for word in lower_words)),
            ]
        )
    return np.asarray(rows, dtype=float)


def load_raw_clean_dataset(path: Path) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
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
    clean_headlines = [build_model_text(headline, "", text_mode="headline") for headline in raw_headlines]
    work = pd.DataFrame({"raw": raw_headlines, "clean": clean_headlines, "label": labels})
    input_examples = len(work)
    blank_examples = int((work["clean"] == "").sum())
    work = work[work["clean"] != ""].copy()

    conflicting = (
        work.groupby("clean")["label"]
        .nunique()
        .loc[lambda series: series > 1]
        .index
    )
    conflict_examples = int(work["clean"].isin(conflicting).sum())
    if len(conflicting) > 0:
        work = work[~work["clean"].isin(conflicting)].copy()

    duplicate_examples = int(work.duplicated("clean").sum())
    work = work.drop_duplicates("clean", keep="first").copy()
    output_labels = work["label"].tolist()
    summary = {
        "input_examples": input_examples,
        "used_examples": len(work),
        "blank_examples": blank_examples,
        "dropped_blank_examples": blank_examples,
        "conflicting_duplicate_examples": conflict_examples,
        "duplicate_headline_examples": duplicate_examples,
        "dropped_duplicate_headline_examples": duplicate_examples,
        "label_counts": {label: output_labels.count(label) for label in LABELS},
        "text_mode": "headline",
    }
    return work["raw"].tolist(), work["clean"].tolist(), output_labels, summary


def make_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=WORD_NGRAM_RANGE,
        max_features=WORD_MAX_FEATURES,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w\w+\b",
    )


def make_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM_RANGE,
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=True,
    )


def fit_transforms(
    clean_texts: list[str],
    raw_texts: list[str],
) -> tuple[TfidfVectorizer, TfidfVectorizer, StandardScaler, Any]:
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


def cross_validate(raw_texts: list[str], clean_texts: list[str], labels: list[str]) -> dict[str, Any]:
    labels_array = np.asarray(labels)
    oof_scores = np.zeros(len(labels), dtype=float)
    fold_default_accuracies: list[float] = []
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for train_idx, validation_idx in skf.split(clean_texts, labels):
        train_raw_texts = [raw_texts[index] for index in train_idx]
        validation_raw_texts = [raw_texts[index] for index in validation_idx]
        train_clean_texts = [clean_texts[index] for index in train_idx]
        validation_clean_texts = [clean_texts[index] for index in validation_idx]
        train_labels = labels_array[train_idx].tolist()
        validation_labels = labels_array[validation_idx].tolist()

        word_vectorizer, char_vectorizer, meta_scaler, X_train = fit_transforms(
            train_clean_texts,
            train_raw_texts,
        )
        X_validation = transform_features(
            validation_clean_texts,
            validation_raw_texts,
            word_vectorizer,
            char_vectorizer,
            meta_scaler,
        )

        clf = LinearSVC(
            C=SVC_C,
            class_weight="balanced",
            max_iter=20000,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, train_labels)
        scores = positive_scores(clf, X_validation)
        oof_scores[validation_idx] = scores
        fold_default_accuracies.append(
            float(accuracy_score(validation_labels, predict_with_threshold(scores, 0.0)))
        )

    threshold, threshold_accuracy = tune_threshold(oof_scores, labels)
    default_preds = predict_with_threshold(oof_scores, 0.0)
    threshold_preds = predict_with_threshold(oof_scores, threshold)
    binary_targets = np.asarray([1 if label == LABELS[1] else 0 for label in labels], dtype=int)

    return {
        "model": "headline_tfidf_meta_linearsvc",
        "classifier": {"kind": "linearsvc", "c": SVC_C, "class_weight": "balanced"},
        "feature_config": {
            "word_max_features": WORD_MAX_FEATURES,
            "char_max_features": CHAR_MAX_FEATURES,
            "meta_features": int(headline_meta_features(["example headline"]).shape[1]),
            "word_ngram_range": list(WORD_NGRAM_RANGE),
            "char_ngram_range": list(CHAR_NGRAM_RANGE),
            "text_mode": "headline",
        },
        "fold_default_accuracy_mean": float(np.mean(fold_default_accuracies)),
        "fold_default_accuracy_std": float(np.std(fold_default_accuracies)),
        "oof_default_accuracy": float(accuracy_score(labels, default_preds)),
        "oof_threshold_accuracy": float(threshold_accuracy),
        "oof_default_macro_f1": float(f1_score(labels, default_preds, labels=LABELS, average="macro")),
        "oof_threshold_macro_f1": float(
            f1_score(labels, threshold_preds, labels=LABELS, average="macro")
        ),
        "oof_threshold_weighted_f1": float(
            f1_score(labels, threshold_preds, labels=LABELS, average="weighted")
        ),
        "oof_roc_auc": float(roc_auc_score(binary_targets, oof_scores)),
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


def encode_payload(payload: dict[str, Any]) -> str:
    return base64.b64encode(
        zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), level=9)
    ).decode("ascii")


def build_preprocess_py(
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    meta_scaler: StandardScaler,
) -> str:
    word_terms = terms_by_index(word_vectorizer)
    char_terms = terms_by_index(char_vectorizer)
    payload = {
        "word_terms": word_terms,
        "word_idf": [float(value) for value in word_vectorizer.idf_],
        "char_terms": char_terms,
        "char_idf": [float(value) for value in char_vectorizer.idf_],
        "meta_scale": [float(value) if float(value) != 0.0 else 1.0 for value in meta_scaler.scale_],
    }
    encoded = encode_payload(payload)
    wrapped = "\n".join(f'    "{chunk}"' for chunk in textwrap.wrap(encoded, width=88))
    word_min, word_max = WORD_NGRAM_RANGE
    char_min, char_max = CHAR_NGRAM_RANGE

    return f'''from __future__ import annotations

import base64
import html
import json
import math
import re
import zlib
from typing import Dict, List, Tuple

import pandas as pd
import torch

NUM_WORD_FEATURES = {len(word_terms)}
NUM_CHAR_FEATURES = {len(char_terms)}
NUM_META_FEATURES = {len(meta_scaler.scale_)}
NUM_FEATURES = {len(word_terms) + len(char_terms) + 10}
WORD_TOKEN_PATTERN = re.compile(r"(?u)\\b\\w\\w+\\b")
WORD_NGRAM_RANGE = ({word_min}, {word_max})
CHAR_NGRAM_RANGE = ({char_min}, {char_max})
SUBLINEAR_TF = True
HEADLINE_COLUMNS = ["headline", "scraped_headline", "alternative_headline", "title"]

_VECTORIZER_B64 = (
{wrapped}
)
_WORD_VOCAB: Dict[str, int] | None = None
_WORD_IDF: List[float] | None = None
_CHAR_VOCAB: Dict[str, int] | None = None
_CHAR_IDF: List[float] | None = None
_META_SCALE: List[float] | None = None


def _load_vectorizers() -> Tuple[Dict[str, int], List[float], Dict[str, int], List[float], List[float]]:
    global _WORD_VOCAB, _WORD_IDF, _CHAR_VOCAB, _CHAR_IDF, _META_SCALE
    if (
        _WORD_VOCAB is None
        or _WORD_IDF is None
        or _CHAR_VOCAB is None
        or _CHAR_IDF is None
        or _META_SCALE is None
    ):
        payload = json.loads(zlib.decompress(base64.b64decode(_VECTORIZER_B64)).decode("utf-8"))
        _WORD_VOCAB = {{term: index for index, term in enumerate(payload["word_terms"])}}
        _WORD_IDF = [float(value) for value in payload["word_idf"]]
        _CHAR_VOCAB = {{term: index for index, term in enumerate(payload["char_terms"])}}
        _CHAR_IDF = [float(value) for value in payload["char_idf"]]
        _META_SCALE = [float(value) if float(value) != 0.0 else 1.0 for value in payload["meta_scale"]]
    return _WORD_VOCAB, _WORD_IDF, _CHAR_VOCAB, _CHAR_IDF, _META_SCALE


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


def build_model_text(headline: str, url: str = "") -> str:
    return clean_text(headline)


SOURCE_SUFFIX_PATTERN = re.compile(
    r"\\s+[-|:]\\s+"
    r"(fox news|fox business|nbc news|nbc news now|nbc select|msnbc|today)"
    r"\\s*$",
    flags=re.IGNORECASE,
)
POLITICS_WORDS = {sorted(POLITICS_WORDS)!r}
LOCATION_WORDS = {sorted(LOCATION_WORDS)!r}
SOFT_ATTRIBUTION_WORDS = {sorted(SOFT_ATTRIBUTION_WORDS)!r}
APPROX_WORDS = {sorted(APPROX_WORDS)!r}
ACTION_VERBS = {sorted(ACTION_VERBS)!r}
OPINION_WORDS = {sorted(OPINION_WORDS)!r}
BE_FORMS = r"{BE_FORMS}"


def strip_source_suffix(text: str) -> str:
    text = html.unescape(str(text)).strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\u00a0", " ")
    text = text.replace("\\u2019", "'").replace("\\u2018", "'")
    text = text.replace("\\u201c", '"').replace("\\u201d", '"')
    text = text.replace("\\u2013", "-").replace("\\u2014", "-")
    text = SOURCE_SUFFIX_PATTERN.sub("", text)
    return re.sub(r"\\s+", " ", text).strip()


def contains_any(text: str, terms: List[str]) -> float:
    return float(any(term in text for term in terms))


def count_any(text: str, terms: List[str]) -> float:
    return float(sum(text.count(term) for term in terms))


def headline_meta_features(text: str, meta_scale: List[float]) -> torch.Tensor:
    raw = strip_source_suffix(text)
    lower = raw.lower()
    chars = len(raw)
    words = re.findall(r"\\b\\w+\\b", raw)
    lower_words = re.findall(r"\\b\\w+\\b", lower)
    letters = sum(1 for char in raw if char.isalpha())
    uppercase = sum(1 for char in raw if char.isupper())
    punctuation = sum(1 for char in raw if char in "!?;:'\\".,-$%&()[]")
    titlecase_words = sum(1 for word in words if word[:1].isupper() and word[1:].islower())
    allcaps_words = sum(1 for word in words if len(word) > 1 and word.isupper())
    person_names = re.findall(r"\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+\\b", raw)
    passive = re.search(rf"\\b{{BE_FORMS}}\\s+\\w+(ed|en)\\b", lower)
    exact_number = bool(re.search(r"\\b\\d{{2,}}\\b", raw))
    percent_or_money = "%" in raw or "$" in raw
    starts_with_name = bool(re.match(r"^[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+\\b", raw))
    starts_with_quote = raw.startswith(('"', "'"))
    starts_with_number = bool(re.match(r"^\\d", raw))
    word_count = len(words)
    values = [
        math.log1p(chars),
        math.log1p(word_count),
        sum(len(word) for word in words) / max(word_count, 1),
        uppercase / max(letters, 1),
        titlecase_words / max(word_count, 1),
        allcaps_words / max(word_count, 1),
        punctuation / max(chars, 1),
        raw.count("?") / max(word_count, 1),
        raw.count("!") / max(word_count, 1),
        raw.count("'") / max(word_count, 1),
        raw.count('"') / max(word_count, 1),
        raw.count(":") / max(word_count, 1),
        raw.count(",") / max(word_count, 1),
        raw.count(".") / max(word_count, 1),
        raw.count("-") / max(word_count, 1),
        float(any(char.isdigit() for char in raw)),
        sum(char.isdigit() for char in raw) / max(chars, 1),
        float(exact_number),
        float(percent_or_money),
        count_any(lower, APPROX_WORDS),
        float(passive is not None),
        count_any(lower, ACTION_VERBS),
        float('"' in raw or "'" in raw),
        float(starts_with_name),
        min(len(person_names), 3) / 3.0,
        float(starts_with_quote),
        float(starts_with_number),
        contains_any(lower, POLITICS_WORDS),
        count_any(lower, POLITICS_WORDS) / max(word_count, 1),
        contains_any(lower, LOCATION_WORDS),
        count_any(lower, LOCATION_WORDS) / max(word_count, 1),
        contains_any(lower, SOFT_ATTRIBUTION_WORDS),
        count_any(lower, OPINION_WORDS),
        float(any(word.endswith("ing") for word in lower_words)),
        float(any(word.endswith("ed") for word in lower_words)),
        float(any(word in {{"best", "worst", "top", "most", "least", "more", "less"}} for word in lower_words)),
    ]
    scaled = [value / scale for value, scale in zip(values, meta_scale)]
    return torch.tensor(scaled, dtype=torch.float32)


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
    word_vocab, word_idf, char_vocab, char_idf, meta_scale = _load_vectorizers()
    raw = strip_source_suffix(headline)
    cleaned = build_model_text(raw)
    word_counts: Dict[int, float] = {{}}
    char_counts: Dict[int, float] = {{}}

    for term in iter_word_ngrams(tokenize_words(cleaned)):
        index = word_vocab.get(term)
        if index is not None:
            word_counts[index] = word_counts.get(index, 0.0) + 1.0

    for term in iter_char_wb_ngrams(cleaned):
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

    return torch.cat([word_features, char_features, headline_meta_features(raw, meta_scale)])


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
    X = [vectorize_headline(text) for text in headlines]
    return X, labels
'''


def export_final(
    raw_texts: list[str],
    clean_texts: list[str],
    labels: list[str],
    threshold: float,
) -> dict[str, Any]:
    word_vectorizer, char_vectorizer, meta_scaler, X_all = fit_transforms(clean_texts, raw_texts)
    clf = LinearSVC(C=SVC_C, class_weight="balanced", max_iter=20000, random_state=RANDOM_STATE)
    clf.fit(X_all, labels)
    coef, intercept = positive_coef_intercept(clf)

    num_features = int(X_all.shape[1])
    weights = torch.zeros((len(LABELS), num_features), dtype=torch.float32)
    bias = torch.zeros(len(LABELS), dtype=torch.float32)
    positive_index = LABELS.index(LABELS[1])
    weights[positive_index] = torch.tensor(coef, dtype=torch.float32)
    bias[positive_index] = torch.tensor(intercept - threshold, dtype=torch.float32)

    torch.save({"linear.weight": weights, "linear.bias": bias}, WEIGHTS_PATH)
    MODEL_PATH.write_text(build_model_py(num_features), encoding="utf-8")
    PREPROCESS_PATH.write_text(
        build_preprocess_py(word_vectorizer, char_vectorizer, meta_scaler),
        encoding="utf-8",
    )
    return {
        "num_features": num_features,
        "threshold_exported_as_bias_shift": threshold,
        "weights_path": str(WEIGHTS_PATH),
        "model_path": str(MODEL_PATH),
        "preprocess_path": str(PREPROCESS_PATH),
    }


def build_summary(metrics: dict[str, Any]) -> str:
    best = metrics["best_model"]
    matrix = best["confusion_matrix"]
    report = best["classification_report"]
    return f"""# Headline-Only Meta Model Summary

## Final Selected Model

- Model: LinearSVC exported as a PyTorch linear layer
- Text mode: headline only
- TF-IDF: word 1-3 grams ({WORD_MAX_FEATURES} features) + character 2-5 grams ({CHAR_MAX_FEATURES} features)
- Extra headline-only metadata features: {best["feature_config"]["meta_features"]}
- C: {SVC_C}
- Threshold: {best["threshold"]:.6f}

## 5-Fold Cross-Validated Results

- Accuracy: {best["oof_threshold_accuracy"]:.4f}
- Macro F1: {best["oof_threshold_macro_f1"]:.4f}
- Weighted F1: {best["oof_threshold_weighted_f1"]:.4f}
- ROC-AUC: {best["oof_roc_auc"]:.4f}
- FoxNews precision/recall/F1: {report["FoxNews"]["precision"]:.4f} / {report["FoxNews"]["recall"]:.4f} / {report["FoxNews"]["f1-score"]:.4f}
- NBC precision/recall/F1: {report["NBC"]["precision"]:.4f} / {report["NBC"]["recall"]:.4f} / {report["NBC"]["f1-score"]:.4f}

## Confusion Matrix

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | {matrix[0][0]} | {matrix[0][1]} |
| NBC | {matrix[1][0]} | {matrix[1][1]} |

## Notes

The final model does not use URL slug/path/domain features. The URL column is used only to infer local labels when a CSV does not provide a `label` or `source` column.
"""


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_texts, clean_texts, labels, dataset_summary = load_raw_clean_dataset(DATA_PATH)
    best_model = cross_validate(raw_texts, clean_texts, labels)
    export_summary = export_final(raw_texts, clean_texts, labels, best_model["threshold"])
    metrics = {
        "dataset_path": str(DATA_PATH),
        "dataset": dataset_summary,
        "validation_protocol": "stratified_cv",
        "selection_metric": "oof_threshold_accuracy",
        "best_model": best_model,
        "final_export": export_summary,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(build_summary(metrics), encoding="utf-8")
    print(json.dumps(best_model, indent=2))
    print(f"saved weights to {WEIGHTS_PATH}")
    print(f"rewrote model constants in {MODEL_PATH}")
    print(f"rewrote preprocess constants in {PREPROCESS_PATH}")
    print(f"saved metrics to {METRICS_PATH}")
    print(f"saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
