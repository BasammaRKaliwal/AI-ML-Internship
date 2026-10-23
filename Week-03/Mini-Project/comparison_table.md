| Model | Parameters | Augmentation | Epochs | Validation accuracy | Test accuracy | Wrong (of 10,000) | Training time |
|---|---|---|---|---|---|---|---|
| Fully connected | 235,146 | No | 10 | 89.52% | 88.37% | 1163 | 173 s |
| Small CNN | 105,866 | No | 8 | 91.17% | 90.88% | 912 | 233 s |
| Final CNN | 241,994 | Yes | 10 | 91.58% | 90.98% | 902 | 819 s |
