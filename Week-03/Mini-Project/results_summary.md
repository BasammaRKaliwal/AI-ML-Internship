## Results

| Model | Parameters | Augmentation | Epochs | Validation accuracy | Test accuracy | Wrong (of 10,000) | Training time |
|---|---|---|---|---|---|---|---|
| Fully connected | 235,146 | No | 10 | 89.52% | 88.37% | 1163 | 173 s |
| Small CNN | 105,866 | No | 8 | 91.17% | 90.88% | 912 | 237 s |
| Final CNN | 241,994 | Yes | 10 | 91.58% | 90.98% | 902 | 767 s |

- The final CNN reached **90.98%** test accuracy (902 wrong out of 10000).
  That is +2.61 points against the fully connected baseline and +0.10 points against the small CNN.
- Easiest category: Sandal (97.8%). Hardest: Shirt (75.6%).
- Most common mistake: Shirt predicted as T-shirt/top (128 times).
- Saved model: `fashion_cnn.pth` (reloaded and checked: identical accuracy).
