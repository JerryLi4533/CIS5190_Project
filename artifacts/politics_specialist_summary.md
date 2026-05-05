# Politics Specialist Summary

## Dataset

- Politics examples: 1322
- FoxNews politics examples: 754
- NBC politics examples: 568
- Mojibake repair: enabled for this experiment

## Accuracy Comparison

| Model | Scope | Accuracy | Macro F1 | ROC-AUC | Threshold |
| --- | --- | ---: | ---: | ---: | ---: |
| Current-style general OOF | politics only | 0.8086 | 0.8058 | 0.8875 | -0.043379 |
| Politics specialist OOF | politics only | 0.7958 | 0.7948 | 0.8726 | -0.118898 |
| Routed OOF | all examples | 0.8347 | 0.8347 | 0.9137 | mixed |

## Politics Specialist Confusion Matrix

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | 571 | 183 |
| NBC | 87 | 481 |

## Highest-Confidence Politics Specialist Errors

- true=NBC, pred=FoxNews, margin=1.467: 'Lock him up!': Hillary Clinton smiles and nods amid chants echoing Trump supporters
- true=FoxNews, pred=NBC, margin=1.157: 'Gutfeld!' draws largest audience in program history with Trump appearance
- true=FoxNews, pred=NBC, margin=1.116: Biden’s moral equivalency between Israel and the Palestinians will result in failure again
- true=FoxNews, pred=NBC, margin=1.070: Biden pledges $7.3B in 'clean energy' spending with national debt at $35T
- true=FoxNews, pred=NBC, margin=1.046: Zelenskyy downplays comment that Trump doesn't know how to end Russia's war with Ukraine
- true=FoxNews, pred=NBC, margin=1.045: Republicans 'skeptical' of DOJ move to block Russian election interference
- true=FoxNews, pred=NBC, margin=1.018: Pence declines to endorse Trump, won't back Biden
- true=FoxNews, pred=NBC, margin=1.007: Bill Maher: Trump is 'right' about the dangerous rhetoric aimed at him, 'but he is a threat to democracy!'
- true=FoxNews, pred=NBC, margin=0.996: Local residents explode at Biden officials over plan to release grizzly bears near their communities
- true=NBC, pred=FoxNews, margin=0.933: Inside the poll numbers: Which key groups have moved and which ones haven't since Biden's exit
- true=FoxNews, pred=NBC, margin=0.930: The unnoticed election that could determine the future
- true=FoxNews, pred=NBC, margin=0.922: Biden is clearly in poor health. We deserve an honest and transparent report
