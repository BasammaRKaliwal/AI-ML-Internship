# Mini Project 3: Fashion Item Recognition

**Notebook:** [mini_project3_fashion_item_recognition.ipynb](mini_project3_fashion_item_recognition.ipynb)
**Saved model:** `fashion_cnn.pth`

## Goal
Build a clothing classifier (shirts, trousers, bags, footwear and more) and compare three models.

## How it works
1. **Data:** Fashion-MNIST, split 54,000 train / 6,000 validation / 10,000 test. The training copy uses light augmentation (random flip and small shifts); validation and test images are never changed.
2. **Three models**
   - Fully connected network (baseline)
   - Small CNN (2 conv layers)
   - Final CNN: three Conv + BatchNorm + ReLU + MaxPool blocks (32, 64, 128 filters), then Dropout and a classifier head
3. **Model choice:** the best model is picked by **validation** accuracy; the test set is used only once for the final numbers.
4. **Outputs:** comparison table, prediction gallery, per-category gallery, confusion matrix, most confident mistakes, and the saved model (reloaded and checked to give identical accuracy).

## Comparison table
PASTE the contents of `comparison_table.md` here.

## Results
PASTE the contents of `results_summary.md` here.

![Prediction gallery](prediction_gallery.png)
![Per-category gallery](category_gallery.png)
![Confusion matrix](confusion_matrix.png)

## Error analysis and conclusion
WRITE your own explanation here.

## Run it
```
pip install torch torchvision numpy matplotlib
jupyter nbconvert --to notebook --execute --inplace mini_project3_fashion_item_recognition.ipynb
```
