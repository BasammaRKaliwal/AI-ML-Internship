# Mini Project 2: Handwritten Digit Classifier

**Notebook:** [mini_project2_digit_classifier.ipynb](mini_project2_digit_classifier.ipynb)

## Goal
Classify handwritten digits 0 to 9 from the MNIST dataset using a fully connected neural network, then study where and why it makes mistakes.

## Data
MNIST: 60,000 training and 10,000 test images (28x28 grayscale). 6,000 training images are held out for validation. Pixels are normalised with the standard MNIST mean (0.1307) and standard deviation (0.3081).

## Model
Each image is flattened into 784 numbers and passed through **784 → 256 → 128 → 10** with ReLU and dropout (0.2). It has 235,146 parameters.

## Training
Adam optimizer (learning rate 0.001), cross-entropy loss, batch size 128, 10 epochs. The weights with the best validation accuracy are kept.

## Results
- Best validation accuracy: 97.88%
- **Test accuracy: 97.92%** (208 wrong out of 10,000)
- Most common mistakes: 4 → 9 (17 times), 9 → 4 (11), 7 → 2 (11), 8 → 5 (8), 9 → 5 (7), 8 → 0 (7)
- In 75% of wrong predictions, the correct digit was the model's second choice.

## Error analysis
The notebook shows the 10 most confident wrong predictions. The causes are similar shapes (4 and 9), messy or unusual handwriting, and the fact that a fully connected network flattens the image and ignores which pixels are neighbours. A convolutional network or data augmentation could improve the result.

## Files
- `mini_project2_digit_classifier.ipynb`: code, metrics and written error analysis
- `training_curves.png`: loss and accuracy per epoch
- `confusion_matrix.png`: test-set confusion matrix
- `sample_predictions.png`: sample predictions with true and predicted labels
- `misclassified.png`: the 10 most confident mistakes

## How to run
Install the libraries with `pip install torch torchvision numpy matplotlib`, open the notebook in VS Code, select a Python kernel and click **Run All**. The first run downloads MNIST (about 11 MB), and training takes a few minutes on a laptop CPU.