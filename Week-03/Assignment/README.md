# Assignment 3: Build and Evaluate a CNN

**Notebook:** [assignment3_cnn_fashion_mnist.ipynb](assignment3_cnn_fashion_mnist.ipynb)

## Goal
Show that a small convolutional neural network (CNN) recognises clothing images better than a fully connected network, and understand where it still fails.

## Data
Fashion-MNIST: 70,000 grayscale images of 28 x 28 pixels in 10 clothing categories (60,000 for training, 10,000 for testing). I set aside 6,000 training images as a validation set. Pixels are normalised with the mean and standard deviation of the training images only.

## How it works
1. **Load and visualise** the images, and check that the classes are balanced.
2. **Convolution demo:** a hand-made edge filter slid over an image shows what a convolution, padding and max pooling do.
3. **Two models** with the same optimiser, learning rate and epochs:
   - Fully connected baseline: 784 -> 256 -> 128 -> 10. It flattens the image, so it ignores which pixels are neighbours.
   - Small CNN: 2 x (Conv 3x3 + ReLU + MaxPool), then Linear 64 -> 10. Filters look at small neighbourhoods and reuse the same weights everywhere, so it needs fewer parameters and finds edges and shapes.
4. **Training and validation curves** for both models. A growing gap between them means overfitting.
5. **Evaluation on the test set:** accuracy, per-class recall, confusion matrix.
6. **Error analysis:** the five most confident mistakes and why the model confused them.

## Results
PASTE the contents of `results_summary.md` here (the notebook creates it).

![Training curves](training_curves.png)
![Confusion matrix](confusion_matrix.png)
![Misclassified examples](misclassified.png)

## Five misclassified examples
PASTE your own explanation from the notebook here.

## Conclusion
WRITE two or three sentences: which model won, by how much, and which categories are still hard.
