# Politics Feature Experiment Summary

## Overall OOF Results

| Model | Accuracy | Macro F1 | ROC-AUC | Errors | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline 10 meta + mojibake repair | 0.8392 | 0.8390 | 0.9176 | 611 | -0.043379 |
| Enhanced politics meta | 0.8373 | 0.8367 | 0.9161 | 618 | 0.039976 |

## Politics Subset Results

| Model | Politics Accuracy | Macro F1 | ROC-AUC | Errors |
| --- | ---: | ---: | ---: | ---: |
| Baseline 10 meta + mojibake repair | 0.8086 | 0.8058 | 0.8875 | 253 |
| Enhanced politics meta | 0.7995 | 0.7943 | 0.8834 | 265 |

## Enhanced Overall Confusion Matrix

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | 1705 | 295 |
| NBC | 323 | 1476 |

## Highest-Confidence Enhanced Errors

- true=FoxNews, pred=NBC, margin=2.056: White House says Trump’s tariffs will destroy manufacturing, exacerbate inflation
- true=NBC, pred=FoxNews, margin=1.995: Schiff's powerful closing speech: 'Is there one among you who will say, Enough!'?
- true=FoxNews, pred=NBC, margin=1.706: How to score cheap stuff (to keep or resell)
- true=NBC, pred=FoxNews, margin=1.543: Trump says presidential civilian award is 'better' than top military honor whose recipients are 'dead' or 'hit' by bullets
- true=NBC, pred=FoxNews, margin=1.422: Chuck Schumer appeals to Harris to attend Al Smith dinner at request of prominent Cardinal
- true=NBC, pred=FoxNews, margin=1.398: Kamala Harris defends her policy changes in first interview: 'My values have not changed'
- true=FoxNews, pred=NBC, margin=1.241: Pence declines to endorse Trump, won't back Biden
- true=NBC, pred=FoxNews, margin=1.181: 2024 Olympics: Simone Biles takes gold in vault, Sha'Carri Richardson gets silver in 100 meter
- true=NBC, pred=FoxNews, margin=1.175: 'I'm not a progressive': Fetterman breaks with the left, showing a maverick side
- true=NBC, pred=FoxNews, margin=1.159: 'God help us': Displaced Gazans who fled bombardment now face health crisis in a makeshift tent city
- true=FoxNews, pred=NBC, margin=1.127: My parents were kidnapped by Hamas. They are not a footnote to Gaza war, they are its essence
- true=NBC, pred=FoxNews, margin=1.103: 'We are the underdogs': Harris introduces running mate Walz as a coach, veteran and protector of reproductive rights
