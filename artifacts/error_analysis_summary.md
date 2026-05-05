# Error Analysis Summary

## OOF Metrics

- Accuracy: 0.8392
- Macro F1: 0.8390
- ROC-AUC: 0.9176
- Threshold: -0.043379
- Total errors: 611 / 3799

## Confusion Matrix

| True label | Pred FoxNews | Pred NBC |
| --- | ---: | ---: |
| FoxNews | 1650 | 350 |
| NBC | 261 | 1538 |

## Error Direction

- FoxNews predicted as NBC: 350
- NBC predicted as FoxNews: 261

## Error Topics

- politics: 253
- other: 237
- world: 54
- crime: 36
- health: 17
- culture: 9
- business: 5

## Highest-Confidence Errors

- true=FoxNews, pred=NBC, margin=1.800: How to score cheap stuff (to keep or resell)
- true=NBC, pred=FoxNews, margin=1.630: Schiff's powerful closing speech: 'Is there one among you who will say, Enough!'?
- true=NBC, pred=FoxNews, margin=1.564: Trump says presidential civilian award is 'better' than top military honor whose recipients are 'dead' or 'hit' by bullets
- true=FoxNews, pred=NBC, margin=1.331: Pence declines to endorse Trump, won't back Biden
- true=NBC, pred=FoxNews, margin=1.297: Chuck Schumer appeals to Harris to attend Al Smith dinner at request of prominent Cardinal
- true=NBC, pred=FoxNews, margin=1.257: 'God help us': Displaced Gazans who fled bombardment now face health crisis in a makeshift tent city
- true=FoxNews, pred=NBC, margin=1.194: Democratic heavyweights to speak at party's convention; preparing for large Palestinian protests
- true=NBC, pred=FoxNews, margin=1.183: 2024 Olympics: Simone Biles takes gold in vault, Sha'Carri Richardson gets silver in 100 meter
- true=FoxNews, pred=NBC, margin=1.103: My parents were kidnapped by Hamas. They are not a footnote to Gaza war, they are its essence
- true=FoxNews, pred=NBC, margin=1.103: Meet the Hurricane Milton babies born at Florida hospitals during the storm
- true=FoxNews, pred=NBC, margin=1.083: Turkey's Erdogan threatens to invade Israel over war in Gaza as regional tensions grow
- true=FoxNews, pred=NBC, margin=1.079: Biden’s moral equivalency between Israel and the Palestinians will result in failure again
