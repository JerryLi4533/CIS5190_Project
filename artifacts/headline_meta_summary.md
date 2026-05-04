# Headline-Only Meta Model Summary

## Final Selected Model

- Model: LinearSVC exported as a PyTorch linear layer
- Text mode: headline only
- TF-IDF: word 1-3 grams (30000 features) + character 2-5 grams (50000 features)
- Extra headline-only metadata features: 36
- C: 0.8
- Threshold: -0.036013

## 5-Fold Cross-Validated Results

- Accuracy: 0.8373
- Macro F1: 0.8371
- Weighted F1: 0.8374
- ROC-AUC: 0.9151
- FoxNews precision/recall/F1: 0.8558 / 0.8310 / 0.8432
- NBC precision/recall/F1: 0.8180 / 0.8444 / 0.8310

## Confusion Matrix

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | 1662 | 338 |
| NBC | 280 | 1519 |

## Notes

The final model does not use URL slug/path/domain features. The URL column is used only to infer local labels when a CSV does not provide a `label` or `source` column.
