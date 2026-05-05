# Political Headline Error Analysis

## Motivation

After inspecting the current model's out-of-fold errors, political headlines stood out as the weakest topic group. The overall model performs reasonably well, but political headlines are harder because Fox News and NBC often cover the same candidates, institutions, elections, court decisions, and foreign-policy conflicts. Surface-level terms such as `Trump`, `Biden`, `Harris`, `White House`, `election`, and `court` appear frequently in both sources, so a headline-only TF-IDF model has limited signal to separate source identity from shared news content.

## Current Model Context

The current exported model is a headline-only linear ensemble exported as a PyTorch linear layer. Its input features are:

- 30,000 word TF-IDF features using word 1-3 n-grams.
- 50,000 character TF-IDF features using character 2-5 n-grams.
- 10 headline style features, including length, capitalization, punctuation, quote/colon/question/exclamation counts, and digit presence.

The model does not use URL slug, URL path, or source-domain tokens in the current exported version.

## Error Analysis Result

The recreated 5-fold out-of-fold error analysis produced:

| Metric | Value |
| --- | ---: |
| Total examples | 3,799 |
| Total errors | 611 |
| Overall accuracy | 0.8392 |
| Overall macro F1 | 0.8390 |
| Overall ROC-AUC | 0.9176 |

Political headlines were the largest and weakest error category:

| Topic | Examples | Errors | Error Rate |
| --- | ---: | ---: | ---: |
| Politics | 1,322 | 253 | 19.14% |
| Health | 102 | 17 | 16.67% |
| Other | 1,614 | 237 | 14.68% |
| World | 372 | 54 | 14.52% |
| Crime | 269 | 36 | 13.38% |
| Culture | 76 | 9 | 11.84% |
| Business | 44 | 5 | 11.36% |

The political errors were not one-sided:

| Direction | Count |
| --- | ---: |
| FoxNews predicted as NBC | 140 |
| NBC predicted as FoxNews | 113 |

This suggests that the issue is not just a bad global threshold. The model is genuinely confused by political headlines from both outlets.

## Representative High-Confidence Political Errors

Some high-confidence political errors included:

- True NBC, predicted FoxNews: `Trump says presidential civilian award is 'better' than top military honor whose recipients are 'dead' or 'hit' by bullets`
- True FoxNews, predicted NBC: `Pence declines to endorse Trump, won't back Biden`
- True FoxNews, predicted NBC: `Biden's moral equivalency between Israel and the Palestinians will result in failure again`
- True FoxNews, predicted NBC: `Trump visits Atlanta Chick fil A, buys customers chicken and shakes`
- True FoxNews, predicted NBC: `Trump, Biden face tests in final 2024 presidential primaries`
- True NBC, predicted FoxNews: `White House condemns university presidents after contentious congressional hearing on antisemitism`
- True NBC, predicted FoxNews: `'Lock him up!': Hillary Clinton smiles and nods amid chants echoing Trump supporters`

These examples show that the model is often not failing on obscure cases. It is failing on headlines that are highly political, highly topical, and often stylistically plausible for either outlet.

## Experiment 1: Politics Specialist

We tested whether a separate model trained only on political headlines would improve political performance. The specialist used the same general modeling approach: TF-IDF word/character features, 10 headline style features, and a LinearSVC ensemble with `C=0.8` and `C=1.0`.

| Model | Scope | Accuracy | Macro F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| Current-style general model | Politics only | 0.8086 | 0.8058 | 0.8875 |
| Politics specialist | Politics only | 0.7958 | 0.7948 | 0.8726 |
| Routed general + politics specialist | All examples | 0.8347 | 0.8347 | 0.9137 |

The politics specialist made performance worse. This indicates that isolating the politics subset reduced useful training signal and did not create a better decision boundary. The political subset had only 1,322 examples, and many of the hardest headlines require subtle source-style distinctions rather than simple political vocabulary.

## Experiment 2: Extra Political Features

We also tested adding hand-built political metadata features to the general model. These features included counts and indicators for:

- Political people such as Trump, Biden, Harris, Vance, Walz, Pence, and Clinton.
- Institutions such as White House, Congress, Senate, Supreme Court, DOJ, FBI, RNC, and DNC.
- Party terms such as GOP, Republican, Democrat, liberal, conservative, and progressive.
- Election terms such as poll, campaign, voters, ballot, primary, nominee, debate, rally, and swing state.
- Framing verbs such as says, warns, slams, blasts, rips, claims, accuses, defends, and condemns.
- Political phrase patterns such as `person + says`, `campaign says`, `White House says`, quote-after-colon structure, and person-name starts.

The result was also worse:

| Model | Overall Accuracy | Overall Macro F1 | Politics Accuracy | Politics Errors |
| --- | ---: | ---: | ---: | ---: |
| Baseline 10 meta + normalization | 0.8392 | 0.8390 | 0.8086 | 253 |
| Enhanced politics metadata | 0.8373 | 0.8367 | 0.7995 | 265 |

These features likely duplicated information already captured by TF-IDF n-grams. Instead of adding genuinely new signal, they amplified noisy or misleading correlations.

## Encoding and Text Normalization Note

During analysis, some output appeared garbled in PowerShell, such as curly quotes displaying incorrectly. A Unicode-level scan showed that the source files were not broadly corrupted. The issue was mostly terminal display encoding, not broken dataset content.

However, normalizing smart quotes, curly apostrophes, dashes, and unusual punctuation may still be useful for feature stability. This should be treated as text normalization rather than evidence of a damaged dataset.

## Interpretation

The political headline problem appears to be a real limitation of headline-only source classification. Political headlines are difficult because:

- Both outlets cover the same national figures and events.
- The strongest words are topical rather than source-specific.
- Many headlines are written in a neutral wire-service style.
- Some Fox headlines resemble NBC headline style, especially straight-news politics coverage.
- Some NBC headlines contain charged quotes or conflict framing that resemble Fox style.
- Hand-built political features mostly repeat information already available to TF-IDF.

In short, the model is not simply missing a few obvious political keywords. It is already seeing those keywords. The hard part is that those keywords do not reliably identify the source.

## Decision

We decided not to modify the final exported model based on these experiments.

The politics specialist and enhanced political metadata features both reduced performance. Keeping the current model is the safer choice for final submission and Hugging Face upload.

## Future Work

The most promising future directions would be:

- Add more balanced political headlines from both outlets, then validate with a time-based split.
- Try larger TF-IDF capacity or different LinearSVC ensemble values rather than hand-built political features.
- Explore transformer-based semantic features if the deployment setting allows a heavier model.
- Use URL slug/path tokens with source-domain words stripped, but only if allowed by the evaluation setting and leakage risk is acceptable.

The main lesson is that political errors are not a quick feature-engineering fix. They reflect a deeper ambiguity in headline-only source classification.
