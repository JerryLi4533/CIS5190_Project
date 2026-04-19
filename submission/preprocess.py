from __future__ import annotations

import hashlib
import re
from typing import List, Tuple

import pandas as pd
import torch

NUM_FEATURES = 4096
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(clean_text(text))


def infer_label(url: str) -> str:
    url = str(url).lower()
    if "foxnews.com" in url:
        return "FoxNews"
    if "nbcnews.com" in url:
        return "NBC"
    raise ValueError(f"Unable to infer label from url: {url}")


def stable_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % NUM_FEATURES


def vectorize_headline(headline: str) -> torch.Tensor:
    tokens = tokenize(headline)
    features = torch.zeros(NUM_FEATURES, dtype=torch.float32)

    for token in tokens:
        features[stable_hash(f"uni::{token}")] += 1.0

    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]}__{tokens[i + 1]}"
        features[stable_hash(f"bi::{bigram}")] += 1.0

    if features.sum() > 0:
        features /= features.sum()
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
