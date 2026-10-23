## Results

| Model | Parameters | Best validation accuracy | Test accuracy |
|---|---|---|---|
| Fully connected baseline | 235,146 | 89.22% | 88.65% |
| Small CNN | 105,866 | 91.13% | 90.46% |

- The CNN was better: the CNN is +1.81 points on the test set while using 45% as many parameters.
- Easiest category for the CNN: Bag (98.5%). Hardest: Shirt (67.9%).
- Most common CNN mistake: Shirt predicted as Coat.
- Train minus validation accuracy in the last epoch: +3.2 points (fully connected) and +2.2 points (CNN).
