# News Source Classification

Starter repository for the CIS 4190/5190 final project track: classifying whether a news headline comes from Fox News or NBC News.

This repo is set up so the team can:

- train a simple baseline locally
- evaluate it with the course `eval_project_b.py` script
- keep submission-facing files in one place
- share a clean project structure on GitHub

## Repository Layout

```text
news-source-classification/
+-- artifacts/                 # Trained weights and experiment outputs
+-- data/
|   +-- raw/
|       +-- url_with_headlines.csv
+-- reports/                   # Notes, plots, draft report assets
+-- scripts/
|   +-- eval_project_b.py
|   +-- scrape_headlines.py
|   +-- train_baseline.py
+-- submission/
|   +-- __init__.py
|   +-- model.py
|   +-- preprocess.py
+-- .gitignore
+-- requirements.txt
```

## Quick Start

1. Create and activate an environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the baseline:

```powershell
python scripts/train_baseline.py
```

To run the stronger experiment search and export the best compatible submission model:

```powershell
python scripts/train_model_search.py
```

For a faster smoke test before the full search:

```powershell
python scripts/train_model_search.py --quick
```

To refine the current best word-trigram LinearSVC model around nearby `C` values:

```powershell
python scripts/train_model_search.py --refine-word-trigrams-svc --svc-c-values 0.35 0.4 0.45 0.5 0.6 0.75
```

To run a larger focused TF-IDF search:

```powershell
python scripts/train_model_search.py --expanded-tfidf
```

To include URL slug tokens with source domains stripped:

```powershell
python scripts/train_model_search.py --expanded-tfidf --text-mode headline_url_slug
```

To evaluate against a separate pseudo-hidden validation CSV instead of random CV, and export the current leaderboard-oriented model:

```powershell
python scripts/train_model_search.py --expanded-tfidf --text-mode headline_url_slug --validation-csv data/raw/recent_validation.csv --expanded-c-values 1.5
```

To try same-feature LinearSVC ensembles that still export as one evaluator-compatible linear model:

```powershell
python scripts/train_word_trigram_ensemble.py
```

Each search writes detailed metrics to `artifacts/model_search_metrics.json` and a report-ready summary to `artifacts/model_improvement_summary.md`.

The exported preprocessing accepts `headline`, `scraped_headline`, `alternative_headline`, or `title` columns, matching the representative Project B screenshots in the guideline.

Current exported result: 93.64% accuracy on the no-overlap post-January-2026 validation CSV, using URL slug tokens with direct source-domain tokens stripped.

If you want to scrape more article titles from a URL list:

```powershell
python scripts/scrape_headlines.py --input data/raw/your_urls.csv --output data/raw/scraped_headlines.csv
```

4. Run the local evaluator:

```powershell
python scripts/eval_project_b.py --model submission/model.py --preprocess submission/preprocess.py --csv data/raw/url_with_headlines.csv --weights artifacts/model.pt
```

The repo already includes a baseline `artifacts/model.pt`, so teammates can run the evaluator immediately after cloning.

## Current Baseline

The included baseline uses:

- deterministic text cleaning
- hashed bag-of-words features with unigrams and bigrams
- a small PyTorch linear classifier

This is not meant to be the final model. It is just a clean starting point that is compatible with the course submission contract.

If your default `python` does not have PyTorch installed, run the commands with a Python environment that does.

## Team Workflow Suggestion

- Put raw or newly scraped datasets under `data/`
- Keep training and analysis scripts under `scripts/`
- Keep only leaderboard-facing files under `submission/`
- Save plots, notes, and report figures under `reports/`

## Submission Notes

The course evaluator expects:

- `submission/preprocess.py`
- `submission/model.py`
- optionally `artifacts/model.pt`

If you change the feature format in `preprocess.py`, make sure `model.py` is updated to expect the same input shape.
