# Report Notes: Final News Source Classifier

## Goal

Our goal was to improve Project B leaderboard accuracy while keeping the final submission compatible with the course evaluator and the staff clarification that the backend provides `(url, headline)` and that the model can directly use the headline.

## Data Used

We used the provided training file `data/raw/url_with_headlines.csv` to fit the final TF-IDF vectorizer and LinearSVC classifier. The URL column is used only to infer labels during local training/evaluation, not as an input feature to the final model.

We also collected `data/raw/recent_validation.csv` from official Fox News and NBC RSS feeds as an extra stress-test dataset. It was not used to fit the final model weights or vocabulary.

## Preprocessing

The final submitted preprocessing is headline-only:

1. It accepts headline-like columns such as `headline`, `scraped_headline`, `alternative_headline`, and `title`.
2. It lowercases text, unescapes HTML entities, removes HTML tags, normalizes smart quotes/dashes, and collapses whitespace.
3. If multiple headline-like columns are present, it concatenates unique non-empty headline strings.
4. It removes obvious source-leakage suffixes such as `| NBC Select`, `- NBC News`, `- Fox News`, `- Fox Business`, `- MSNBC`, and `- Today`.
5. It adds 10 headline-only style features for length, capitalization, punctuation/标点符号 rate, comma/quote/colon/question/exclamation counts, and digit presence.

The final model does not use URL slug/path tokens and does not use source-domain tokens as features.

## Model Search

We started from the baseline linear classifier and moved to TF-IDF features with linear models because this design is simple, fast, interpretable, and compatible with the course backend. We compared Logistic Regression and LinearSVC, then focused on LinearSVC because it performed better in validation.

The final search varied word n-gram ranges, character n-gram ranges, vocabulary sizes, `min_df`, LinearSVC regularization strength `C`, and the prediction threshold.

The selected final model is:

- classifier: average of two LinearSVC models exported as one PyTorch linear layer
- `C`: 0.8 and 1.0
- class weight: balanced
- word TF-IDF: 1-3 grams, 30,000 features
- character TF-IDF: 2-5 grams, 50,000 features
- headline style features: 10
- total feature dimension: 80,010
- text mode: headline only
- threshold: 0.021191

## Validation Strategy

We used 5-fold stratified cross-validation on the provided dataset for model selection. The decision threshold was tuned using out-of-fold scores only.

## Results

The final headline-only model achieved:

- 5-fold CV accuracy: 0.8421
- macro F1: 0.8416
- ROC-AUC: 0.9175

The local evaluator reports 100% accuracy on `url_with_headlines.csv`, but that is training-set evaluation after fitting on all provided examples. It should only be treated as a smoke test that `model.py`, `preprocess.py`, and `model.pt` load correctly.

## Why We Kept This Model

We kept the headline-only TF-IDF plus LinearSVC ensemble because it follows staff guidance, removes source-leakage tokens, has no runtime dependency on `scikit-learn`, `scipy`, `nltk`, or `transformers`, and runs quickly in the leaderboard backend. The exported submission only needs `pandas` and `torch` at inference time.

## Caveat

Headline-only source classification appears to have a natural ceiling around the mid-to-high 80s on this dataset. URL slug features can improve public leaderboard accuracy, but the final safe version avoids them because staff clarified that the backend provides headlines directly and source-leakage tokens should be removed.
