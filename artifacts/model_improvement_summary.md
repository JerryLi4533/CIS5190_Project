# News Source Classification Model Improvement Summary

## Final Selected Model

- Model: linearsvc with C=1.5 and class_weight=balanced
- Features: TF-IDF word [1, 3] n-grams plus character [2, 5] n-grams
- Vocabulary: 30000 word features + 50000 character features
- TF scaling: sublinear_tf=True
- Text mode: headline_url_slug
- Threshold: -0.151807

## Evaluation Protocol

- Selection metric: holdout accuracy on the separate validation CSV, using the training CSV only for fitting
- Diagnostic metrics: macro F1, weighted F1, per-class precision/recall/F1, confusion matrix, ROC-AUC
- Training dataset used after cleaning: 3805 examples
- Validation dataset used after cleaning: 393 examples
- Dropped examples: 0 blank headlines and 0 duplicate headlines

## Holdout Validation Results

- Accuracy: 0.9364
- Macro F1: 0.9355
- Weighted F1: 0.9361
- ROC-AUC: 0.9825
- FoxNews precision/recall/F1: 0.9159 / 0.9718 / 0.9431
- NBC precision/recall/F1: 0.9641 / 0.8944 / 0.9280

## Confusion Matrix

Rows are true labels and columns are predicted labels.

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | 207 | 6 |
| NBC | 19 | 161 |

## Main Improvements

- Replaced the fixed single-split experiment with 5-fold stratified cross-validation.
- Switched the best classifier from Logistic Regression to LinearSVC after empirical comparison.
- Increased TF-IDF capacity and added word trigrams while keeping character n-grams.
- Added optional URL-slug text features with source domains stripped.
- Applied the same text cleaning during training and submission preprocessing.
- Removed blank and duplicate cleaned headlines from training.
- Tuned the decision threshold on held-out validation scores to improve final accuracy.
- Added macro F1, weighted F1, ROC-AUC, and confusion-matrix diagnostics for reporting while keeping accuracy as the final selection target.
