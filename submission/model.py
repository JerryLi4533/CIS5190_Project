from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn

NUM_FEATURES = 80010
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
