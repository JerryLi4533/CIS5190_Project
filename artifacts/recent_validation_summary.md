# Post-January-2026 Pseudo-Hidden Validation Summary

The course staff clarified that no leaderboard test articles were scraped during or after January 2026. To reduce overlap risk, we created a pseudo-hidden validation set from official Fox News and NBC News RSS feeds using articles published on or after February 1, 2026.

## Dataset

- Source file: `data/raw/recent_validation.csv`
- Total examples: 393
- FoxNews examples: 213
- NBC examples: 180
- Cutoff: February 1, 2026 or later
- Collection method: official Fox/NBC RSS feeds

## Best Headline-Only Model On This Validation Set

- Model: LinearSVC with C=0.3
- Features: TF-IDF word 1-3 n-grams plus character 2-5 n-grams
- Vocabulary: 30,000 word features + 50,000 character features
- Threshold: -0.098744

## Headline-Only Validation Results

- Accuracy: 0.6718
- Macro F1: 0.6709
- Weighted F1: 0.6723
- ROC-AUC: 0.7134

## Final Leaderboard-Oriented Model On This Validation Set

- Model: LinearSVC with C=1.5
- Features: TF-IDF word 1-3 n-grams plus character 2-5 n-grams
- Vocabulary: 30,000 word features + 50,000 character features
- Text mode: headline plus source-stripped URL slug tokens
- Threshold: -0.151807

## Final Validation Results

- Accuracy: 0.9364
- Macro F1: 0.9355
- Weighted F1: 0.9361
- ROC-AUC: 0.9825

## Interpretation

The headline-only score is much lower than random 5-fold CV on the starter data, which suggests substantial distribution shift between the starter headlines and newer post-January-2026 headlines. Adding URL slug tokens gives a large gain while still stripping direct source-domain tokens such as `foxnews` and `nbcnews`. Because the hidden leaderboard test is not post-January-2026, this validation set should be interpreted as a no-overlap robustness check rather than a perfect simulation of the hidden test distribution.
