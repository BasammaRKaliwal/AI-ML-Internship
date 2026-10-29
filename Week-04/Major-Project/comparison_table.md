| Configuration | Input size | Augmentation | Epochs | Parameters | Training time | Best val. accuracy | Test accuracy |
|---|---|---|---|---|---|---|---|
| Baseline CNN | 128 px | No | 15 | 1,072,549 | 320 s | 70.8% | 69.6% |
| Improved CNN | 128 px | Yes | 25 | 390,181 | 1278 s | 76.0% | 74.9% |
| Pretrained MobileNetV2 (final model) | 224 px | Yes | 8 | 2,230,277 | 771 s | 91.1% | 89.7% |
