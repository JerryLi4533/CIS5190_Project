# News Source Classification Model Improvement Summary

## Final Selected Model

- Model: LinearSVC with C=0.45 and class_weight=balanced
- Features: TF-IDF word 1-3 n-grams plus character 3-5 n-grams
- Vocabulary: 20,000 word features + 30,000 character features
- TF scaling: sublinear_tf=True
- Threshold: -0.042281

## Evaluation Protocol

- Selection metric: 5-fold cross-validated accuracy, matching the course leaderboard metric
- Diagnostic metrics: macro F1, weighted F1, per-class precision/recall/F1, confusion matrix, ROC-AUC
- Dataset used after cleaning: 3,799 examples
- Dropped examples: 4 blank headlines and 2 duplicate headlines

## Cross-Validated Results

- Accuracy: 0.8313
- Macro F1: 0.8308
- Weighted F1: 0.8313
- FoxNews precision/recall/F1: 0.8406 / 0.8385 / 0.8395
- NBC precision/recall/F1: 0.8210 / 0.8232 / 0.8221

## Confusion Matrix

Rows are true labels and columns are predicted labels.

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | 1677 | 323 |
| NBC | 318 | 1481 |

## Main Improvements

- Replaced the fixed single-split experiment with 5-fold stratified cross-validation.
- Switched the best classifier from Logistic Regression to LinearSVC after empirical comparison.
- Increased TF-IDF capacity and added word trigrams while keeping character n-grams.
- Applied the same text cleaning during training and submission preprocessing.
- Removed blank and duplicate cleaned headlines from training.
- Tuned the decision threshold on out-of-fold scores to improve final accuracy.
- Added macro F1, weighted F1, ROC-AUC, and confusion-matrix diagnostics for reporting while keeping accuracy as the final selection target.
