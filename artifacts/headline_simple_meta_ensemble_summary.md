# Headline-Only Simple Style Ensemble Summary

- Accuracy: 0.8421
- Macro F1: 0.8416
- ROC-AUC: 0.9175
- Text mode: headline only
- Classifier: averaged LinearSVC C=0.8 and C=1.0, exported as PyTorch linear weights
- Features: word 1-3 TF-IDF, char 2-5 TF-IDF, and 10 headline style features for length, capitalization, punctuation/标点符号, comma/quote/colon/question/exclamation counts, and digit presence.
- No URL slug/path/domain features are used as model input.
