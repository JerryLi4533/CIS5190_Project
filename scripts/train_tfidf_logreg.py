from __future__ import annotations

import base64
import json
from pathlib import Path
import textwrap
import zlib

import pandas as pd
from scipy.sparse import hstack
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
MODEL_PATH = ROOT / "submission" / "model.py"
PREPROCESS_PATH = ROOT / "submission" / "preprocess.py"

LABELS = ["FoxNews", "NBC"]
WORD_MAX_FEATURES = 10000
CHAR_MAX_FEATURES = 15000
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)
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


def make_word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=WORD_NGRAM_RANGE,
        max_features=WORD_MAX_FEATURES,
    )


def make_char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM_RANGE,
        max_features=CHAR_MAX_FEATURES,
    )


def fit_vectorizers(texts: list[str]) -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word_vectorizer = make_word_vectorizer()
    char_vectorizer = make_char_vectorizer()
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


def train_classifier(X, y: list[str]) -> LogisticRegression:
    clf = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
    )
    clf.fit(X, y)
    return clf


def terms_by_index(vectorizer: TfidfVectorizer) -> list[str]:
    terms: list[str | None] = [None] * len(vectorizer.vocabulary_)
    for term, index in vectorizer.vocabulary_.items():
        terms[index] = term
    if any(term is None for term in terms):
        raise RuntimeError("Vectorizer vocabulary has missing feature indices.")
    return [str(term) for term in terms]


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


def encode_payload(payload: dict) -> str:
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
    char_min, char_max = CHAR_NGRAM_RANGE

    return f'''from __future__ import annotations

import base64
import json
import re
import zlib
from typing import Dict, List, Tuple

import pandas as pd
import torch

NUM_WORD_FEATURES = {len(word_terms)}
NUM_CHAR_FEATURES = {len(char_terms)}
NUM_FEATURES = {num_features}
WORD_TOKEN_PATTERN = re.compile(r"(?u)\\b\\w\\w+\\b")
CHAR_NGRAM_RANGE = ({char_min}, {char_max})

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
    text = str(text).strip().lower()
    text = text.replace("\\u2019", "'").replace("\\u2018", "'")
    text = text.replace("\\u201c", '"').replace("\\u201d", '"')
    return text


def tokenize_words(text: str) -> List[str]:
    return WORD_TOKEN_PATTERN.findall(clean_text(text))


def iter_word_ngrams(tokens: List[str]):
    for n in (1, 2):
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
        word_features[index] = count * word_idf[index]
    _normalize(word_features)

    char_features = torch.zeros(NUM_CHAR_FEATURES, dtype=torch.float32)
    for index, count in char_counts.items():
        char_features[index] = count * char_idf[index]
    _normalize(char_features)

    return torch.cat([word_features, char_features])


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

    validation_word_vectorizer, validation_char_vectorizer = fit_vectorizers(train_text)
    X_train = transform_texts(train_text, validation_word_vectorizer, validation_char_vectorizer)
    X_val = transform_texts(val_text, validation_word_vectorizer, validation_char_vectorizer)
    validation_clf = train_classifier(X_train, y_train)
    val_pred = validation_clf.predict(X_val)

    final_word_vectorizer, final_char_vectorizer = fit_vectorizers(texts)
    X_all = transform_texts(texts, final_word_vectorizer, final_char_vectorizer)
    final_clf = train_classifier(X_all, labels)
    num_features = X_all.shape[1]

    torch.save(export_state_dict(final_clf, num_features), WEIGHTS_PATH)
    MODEL_PATH.write_text(build_model_py(num_features), encoding="utf-8")
    PREPROCESS_PATH.write_text(
        build_preprocess_py(final_word_vectorizer, final_char_vectorizer),
        encoding="utf-8",
    )

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
            "features": "TF-IDF word unigrams/bigrams plus character 3-5 n-grams",
            "word_max_features": WORD_MAX_FEATURES,
            "char_max_features": CHAR_MAX_FEATURES,
            "total_features": int(num_features),
            "classifier": "LogisticRegression exported to PyTorch Linear",
            "C": 2.0,
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
    print(f"num_features: {num_features}")
    print(f"saved weights to {WEIGHTS_PATH}")
    print(f"rewrote model constants in {MODEL_PATH}")
    print(f"rewrote preprocess constants in {PREPROCESS_PATH}")
    print(f"saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
