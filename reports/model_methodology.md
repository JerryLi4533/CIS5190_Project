# Report Notes: Final News Source Classifier

## Goal

Our goal was to improve holdout leaderboard accuracy for Project B while keeping the submission compatible with the course evaluator. The leaderboard metric is accuracy, so accuracy was the primary model-selection metric. We also tracked macro F1, weighted F1, ROC-AUC, per-class precision/recall/F1, and confusion matrices to check whether improvements were balanced across FoxNews and NBC.

## Data Used

We used the provided training file `data/raw/url_with_headlines.csv` as the only data used to fit the final TF-IDF vectorizer and LinearSVC classifier. This file contains URLs and headlines, with labels inferred from the source domain during local evaluation.

Based on course staff guidance, no leaderboard articles were scraped during or after January 2026. To create a no-overlap validation check, we also collected a separate post-January-2026 validation file, `data/raw/recent_validation.csv`, from official Fox News and NBC RSS feeds. This validation file was not used to fit the final model weights or vocabulary. It was used only to compare hyperparameters and choose the decision threshold.

## Preprocessing

We improved preprocessing in four ways:

1. Robust headline-column handling. The guideline screenshots suggest hidden/test data may contain columns such as `headline`, `scraped_headline`, `alternative_headline`, or `title`. The final preprocessing accepts all of these columns.
2. Text normalization. We lowercase text, unescape HTML entities, remove HTML tags, normalize smart quotes/dashes, collapse whitespace, and remove trailing source suffixes such as `- Fox News`, `| NBC News`, or `- MSNBC`.
3. Multiple headline fields. If several headline-like columns are present, we concatenate the unique non-empty values. This is label-independent, so the feature text does not use the true class label.
4. URL slug features. In addition to headline text, the final model appends tokens from the URL slug. We strip direct source-domain tokens such as `foxnews`, `fox`, `nbcnews`, `nbc`, `www`, and `com`. We use only the slug text, not the domain.

The URL slug feature was the largest improvement. Headline-only models struggled on the post-January validation set, while headline plus source-stripped slug tokens generalized much better.

## Model Search

We started from the baseline linear classifier and moved to TF-IDF features with linear models because this design is simple, fast, interpretable, and compatible with the course backend. We compared Logistic Regression and LinearSVC, then focused on LinearSVC because it consistently performed better.

The final search varied:

- word n-gram ranges: especially 1-3 and 1-4
- character n-gram ranges: especially 2-5, 3-5, and 3-6
- word vocabulary size: 30,000 to 50,000
- character vocabulary size: 50,000 to 80,000
- `min_df`: 1 and 2
- LinearSVC regularization strength `C`
- decision threshold on model scores

The selected final model is:

- classifier: LinearSVC
- `C`: 1.5
- class weight: balanced
- word TF-IDF: 1-3 grams, 30,000 features
- character TF-IDF: 2-5 grams, 50,000 features
- total feature dimension: 80,000
- text mode: headline plus source-stripped URL slug tokens
- threshold: -0.151807

## Validation Strategy

Random 5-fold cross-validation on the provided training set is useful, but it can overestimate performance if similar headlines or collection artifacts appear across folds. To better approximate hidden-set behavior, we used the separate post-January-2026 validation set as a no-overlap check.

This validation set is not a perfect simulation of the leaderboard, because course staff said the hidden leaderboard data was not scraped during or after January 2026. However, it is useful because it tests whether the model can generalize to newer headlines that are unlikely to overlap with the training set.

## Results

Headline-only expanded TF-IDF with LinearSVC reached about 83.7% cross-validated accuracy on the provided training data. This suggested the text style alone had useful but limited signal.

Adding source-stripped URL slug tokens greatly improved performance:

- 5-fold CV on the provided data: about 98.9% accuracy for the final feature/model family
- post-January validation accuracy after threshold tuning: 93.64%
- post-January validation macro F1: 93.55%
- post-January validation weighted F1: 93.61%
- post-January validation ROC-AUC: 98.25%

The local evaluator reports 100% accuracy on `url_with_headlines.csv`, but that is training-set evaluation and should not be reported as evidence of generalization. We only use it as a smoke test that the exported `model.py`, `preprocess.py`, and `model.pt` load correctly.

## Why We Kept This Model

We kept the TF-IDF plus LinearSVC model instead of a transformer because it is lightweight, fast, compatible with the course submission format, and performed strongly on the no-overlap validation set. It also avoids depending on external model downloads or backend packages that may not exist in the course evaluator.

The most important improvement was not model complexity. It was better use of the available input format, especially robust preprocessing and source-stripped URL slug tokens.

## Important Caveat

The URL slug is metadata from the provided URL, so we should describe it transparently in the report. We do not use the source domain itself, and we explicitly remove source-name tokens from the slug text. Still, this feature is stronger than headline-only text and should be framed as using all available text fields from the provided input rather than as a pure headline-only classifier.
