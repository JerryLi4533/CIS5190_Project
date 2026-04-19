from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.model import Model
from submission.preprocess import prepare_data

DATA_PATH = ROOT / "data" / "raw" / "url_with_headlines.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
WEIGHTS_PATH = ARTIFACTS_DIR / "model.pt"
METRICS_PATH = ARTIFACTS_DIR / "baseline_metrics.json"

LABEL_TO_ID = {"FoxNews": 0, "NBC": 1}


def evaluate(model: Model, X: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(y.tolist(), preds.tolist())
    report = classification_report(
        y.tolist(),
        preds.tolist(),
        target_names=["FoxNews", "NBC"],
        output_dict=True,
        zero_division=0,
    )
    return {"accuracy": accuracy, "report": report}


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X_list, y_labels = prepare_data(str(DATA_PATH))
    X = torch.stack(X_list)
    y = torch.tensor([LABEL_TO_ID[label] for label in y_labels], dtype=torch.long)

    indices = list(range(len(y)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y.tolist(),
    )

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True,
    )

    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_accuracy = -1.0

    for epoch in range(1, 31):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        metrics = evaluate(model, X_val, y_val)
        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = metrics["accuracy"]
        print(f"epoch={epoch:02d} loss={epoch_loss:.4f} val_acc={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, X_val, y_val)
    torch.save(best_state, WEIGHTS_PATH)

    summary = {
        "dataset_path": str(DATA_PATH),
        "num_examples": len(y_labels),
        "train_examples": len(train_idx),
        "val_examples": len(val_idx),
        "best_val_accuracy": final_metrics["accuracy"],
        "classification_report": final_metrics["report"],
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved weights to {WEIGHTS_PATH}")
    print(f"saved metrics to {METRICS_PATH}")
    print(json.dumps({"best_val_accuracy": final_metrics["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
