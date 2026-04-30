from __future__ import annotations

import base64
import json
from pathlib import Path
import textwrap
import zlib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "url_with_headlines.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
WEIGHTS_PATH = ARTIFACTS_DIR / "model.pt"
METRICS_PATH = ARTIFACTS_DIR / "improved_metrics.json"
PREPROCESS_PATH = ROOT / "submission" / "preprocess.py"

LABELS = ["FoxNews", "NBC"]
MAX_FEATURES = 15000
NGRAM_RANGE = (1, 2)
RANDOM_STATE = 42


def infer_label(url: str) -> str:
    url = str(url).lower()
    if "foxnews.com" in url:
        return "FoxNews"
    if "nbcnews.com" in url:
        return "NBC"
    raise ValueError(f"Unable to infer label from url: {url}")


def load_dataset() -> tuple[list[str], list[str]]:
    df = pd.read_csv(DATA_PATH)
    if "headline" not in df.columns:
        raise ValueError("Expected a 'headline' column in the CSV.")
    texts = df["headline"].fillna("").astype(str).tolist()
    if "url" in df.columns:
        labels = [infer_label(url) for url in df["url"].tolist()]
    elif "label" in df.columns:
        labels = df["label"].astype(str).tolist()
    elif "source" in df.columns:
        labels = df["source"].astype(str).tolist()
    else:
        raise ValueError("Expected one of: url, label, source columns.")
    return texts, labels


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
    )


def train_classifier(X, y: list[str]) -> LogisticRegression:
    clf = LogisticRegression(
        C=8.0,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
    )
    clf.fit(X, y)
    return clf


def export_state_dict(clf: LogisticRegression, num_features: int) -> dict[str, torch.Tensor]:
    weights = torch.zeros((len(LABELS), num_features), dtype=torch.float32)
    bias = torch.zeros(len(LABELS), dtype=torch.float32)

    class0, class1 = [str(item) for item in clf.classes_]
    class1_index = LABELS.index(class1)
    weights[class1_index] = torch.tensor(clf.coef_[0], dtype=torch.float32)
    bias[class1_index] = torch.tensor(clf.intercept_[0], dtype=torch.float32)
    LABELS.index(class0)

    return {
        "linear.weight": weights,
        "linear.bias": bias,
    }


def build_preprocess_py(vectorizer: TfidfVectorizer) -> str:
    terms_by_index = [None] * len(vectorizer.vocabulary_)
    for term, index in vectorizer.vocabulary_.items():
        terms_by_index[index] = term
    if any(term is None for term in terms_by_index):
        raise RuntimeError("Vectorizer vocabulary has missing feature indices.")

    payload = {
        "terms": terms_by_index,
        "idf": [float(value) for value in vectorizer.idf_],
    }
    encoded = base64.b64encode(
        zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), level=9)
    ).decode("ascii")
    wrapped = "\n".join(f'    "{chunk}"' for chunk in textwrap.wrap(encoded, width=88))

    return f'''from __future__ import annotations

import base64
import json
import re
import zlib
from typing import Dict, List, Tuple

import pandas as pd
import torch

NUM_FEATURES = {len(terms_by_index)}
TOKEN_PATTERN = re.compile(r"(?u)\\b\\w\\w+\\b")

_VECTORIZER_B64 = (
{wrapped}
)
_VOCAB: Dict[str, int] | None = None
_IDF: List[float] | None = None


def _load_vectorizer() -> Tuple[Dict[str, int], List[float]]:
    global _VOCAB, _IDF
    if _VOCAB is None or _IDF is None:
        payload = json.loads(zlib.decompress(base64.b64decode(_VECTORIZER_B64)).decode("utf-8"))
        _VOCAB = {{term: index for index, term in enumerate(payload["terms"])}}
        _IDF = [float(value) for value in payload["idf"]]
    return _VOCAB, _IDF


def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("\\u2019", "'").replace("\\u2018", "'")
    text = text.replace("\\u201c", '"').replace("\\u201d", '"')
    return text


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(clean_text(text))


def iter_ngrams(tokens: List[str]):
    for n in (1, 2):
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i : i + n])


def infer_label(url: str) -> str:
    url = str(url).lower()
    if "foxnews.com" in url:
        return "FoxNews"
    if "nbcnews.com" in url:
        return "NBC"
    raise ValueError(f"Unable to infer label from url: {{url}}")


def vectorize_headline(headline: str) -> torch.Tensor:
    vocab, idf = _load_vectorizer()
    counts: Dict[int, float] = {{}}
    for term in iter_ngrams(tokenize(headline)):
        index = vocab.get(term)
        if index is not None:
            counts[index] = counts.get(index, 0.0) + 1.0

    features = torch.zeros(NUM_FEATURES, dtype=torch.float32)
    for index, count in counts.items():
        features[index] = count * idf[index]

    norm = torch.linalg.vector_norm(features)
    if norm > 0:
        features /= norm
    return features


def prepare_data(path: str) -> Tuple[List[torch.Tensor], List[str]]:
    df = pd.read_csv(path)
    if "headline" not in df.columns:
        raise ValueError("Expected a 'headline' column in the CSV.")

    if "url" in df.columns:
        labels = [infer_label(url) for url in df["url"].tolist()]
    elif "label" in df.columns:
        labels = df["label"].astype(str).tolist()
    elif "source" in df.columns:
        labels = df["source"].astype(str).tolist()
    else:
        raise ValueError("Expected one of: url, label, source columns.")

    headlines = df["headline"].fillna("").astype(str).tolist()
    X = [vectorize_headline(text) for text in headlines]
    return X, labels
'''


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    texts, labels = load_dataset()

    train_text, val_text, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    validation_vectorizer = make_vectorizer()
    X_train = validation_vectorizer.fit_transform(train_text)
    X_val = validation_vectorizer.transform(val_text)
    validation_clf = train_classifier(X_train, y_train)
    val_pred = validation_clf.predict(X_val)

    final_vectorizer = make_vectorizer()
    X_all = final_vectorizer.fit_transform(texts)
    final_clf = train_classifier(X_all, labels)

    torch.save(export_state_dict(final_clf, len(final_vectorizer.vocabulary_)), WEIGHTS_PATH)
    PREPROCESS_PATH.write_text(build_preprocess_py(final_vectorizer), encoding="utf-8")

    metrics = {
        "dataset_path": str(DATA_PATH),
        "num_examples": len(labels),
        "label_counts": {label: labels.count(label) for label in LABELS},
        "validation_split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "train_examples": len(y_train),
            "val_examples": len(y_val),
        },
        "model": {
            "features": "TF-IDF word unigrams and bigrams",
            "max_features": MAX_FEATURES,
            "classifier": "LogisticRegression exported to PyTorch Linear",
            "C": 8.0,
            "class_weight": "balanced",
        },
        "validation_accuracy": accuracy_score(y_val, val_pred),
        "classification_report": classification_report(
            y_val,
            val_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(y_val, val_pred, labels=LABELS).tolist(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"validation_accuracy: {metrics['validation_accuracy']:.6f}")
    print(f"saved weights to {WEIGHTS_PATH}")
    print(f"rewrote preprocess constants in {PREPROCESS_PATH}")
    print(f"saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
