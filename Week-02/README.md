## Assignment

**Neural Network Comparison.** Notebook: [Assignment/assignment2_neural_network_comparison.ipynb](Assignment/assignment2_neural_network_comparison.ipynb)

**Goal:** compare a linear model with a nonlinear neural network on a synthetic two-class dataset, and test how hidden-layer size and learning rate change the result.

**How it works:**
1. Generate 1,000 points shaped like two interleaving half-circles (noise 0.25), then split them 700 train / 150 validation / 150 test.
2. Build a linear model (one layer) and a neural network (2 → hidden → hidden → 1 with ReLU).
3. Train every model with the Adam optimizer and binary cross-entropy loss for 300 epochs.
4. Try hidden sizes 2, 4, 8, 16, 32, 64 and learning rates 0.001, 0.01, 0.05, 0.1.
5. Pick the best setting using validation accuracy, then report its test accuracy.

**Results:**

| Model | Test accuracy |
|---|---|
| Linear model | 90.0% |
| Best neural network (16 hidden units, learning rate 0.01) | 94.7% |

The neural network won because the two classes are curved and a straight line cannot separate them fully. The weakest setting (2 hidden units, learning rate 0.001) reached only 62.7% validation accuracy, and the largest train/validation gap (3.6 points) appeared at 32 hidden units with learning rate 0.1.

**Files:** `decision_boundaries.png`, `hyperparameter_heatmap.png`, `confusion_matrices.png`. To run it, open the notebook in VS Code and click Run All.

## Mini Project

**Handwritten Digit Classifier.** Notebook: [Mini-Project/mini_project2_digit_classifier.ipynb](Mini-Project/mini_project2_digit_classifier.ipynb)

**Goal:** classify handwritten digits 0 to 9 from the MNIST dataset using a fully connected neural network.

**How it works:**
1. Load MNIST (60,000 training and 10,000 test images) and normalise the pixels. Hold out 6,000 training images for validation.
2. Flatten each 28x28 image into 784 numbers and pass it through the layers 784 → 256 → 128 → 10, with ReLU and dropout (0.2).
3. Train for 10 epochs with the Adam optimizer (learning rate 0.001, batch size 128) and cross-entropy loss. Keep the weights with the best validation accuracy.
4. Evaluate on the test set and study the mistakes.

**Results:** test accuracy 97.92% (208 wrong out of 10,000). The most common mistakes were 4 → 9 (17 times), 9 → 4 (11), 7 → 2 (11), 8 → 5 (8), 9 → 5 (7) and 8 → 0 (7). In 75% of wrong predictions the correct digit was the model's second choice. The error analysis in the notebook explains the causes: similar shapes, messy handwriting and unusual styles.

**Files:** `training_curves.png`, `confusion_matrix.png`, `sample_predictions.png`, `misclassified.png`. To run it, open the notebook in VS Code and click Run All. The first run downloads MNIST.
