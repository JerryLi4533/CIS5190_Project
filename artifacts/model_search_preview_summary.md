# News Source Classification Model Improvement Summary

## Final Selected Model

- Model: linearsvc with C=1.0 and class_weight=balanced
- Features: TF-IDF word [1, 3] n-grams plus character [2, 5] n-grams
- Vocabulary: 30000 word features + 50000 character features
- TF scaling: sublinear_tf=True
- Text mode: headline
- Threshold: -0.009247

## Evaluation Protocol

- Selection metric: 5-fold cross-validated accuracy, matching the course leaderboard metric
- Diagnostic metrics: macro F1, weighted F1, per-class precision/recall/F1, confusion matrix, ROC-AUC
- Training dataset used after cleaning: 3799 examples
- Dropped examples: 4 blank headlines and 2 duplicate headlines

## Cross-Validated Results

- Accuracy: 0.8371
- Macro F1: 0.8362
- Weighted F1: 0.8368
- ROC-AUC: 0.9115
- FoxNews precision/recall/F1: 0.8321 / 0.8650 / 0.8482
- NBC precision/recall/F1: 0.8430 / 0.8060 / 0.8241

## Confusion Matrix

Rows are true labels and columns are predicted labels.

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | 1730 | 270 |
| NBC | 349 | 1450 |

## Main Improvements

- Replaced the fixed single-split experiment with 5-fold stratified cross-validation.
- Switched the best classifier from Logistic Regression to LinearSVC after empirical comparison.
- Increased TF-IDF capacity and added word trigrams while keeping character n-grams.
- Added optional URL-slug text features with source domains stripped.
- Applied the same text cleaning during training and submission preprocessing.
- Removed blank and duplicate cleaned headlines from training.
- Tuned the decision threshold on out-of-fold scores to improve final accuracy.
- Added macro F1, weighted F1, ROC-AUC, and confusion-matrix diagnostics for reporting while keeping accuracy as the final selection target.
