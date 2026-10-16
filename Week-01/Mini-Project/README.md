# Mini Project 1: Numerical Pattern Predictor

**Notebook:** [numerical_pattern_predictor.ipynb](numerical_pattern_predictor.ipynb)

## Goal
Generate numerical data that follows a hidden relationship, train a neural network to learn it without being told the formula, and predict outputs for inputs it has never seen.

## How it works
1. **Generate data:** 1,000 random inputs `x` between -5 and 5. The true output is `y = 2*sin(x) + 0.5*x`, plus a little random noise (0.1). The network never sees this formula.
2. **Split:** 70% train (700), 15% validation (150), 15% test (150). The test set is used only at the end.
3. **Normalise:** inputs and outputs are scaled using the mean and standard deviation of the **training set only**, so no information leaks from validation or test.
4. **Model:** a fully connected network, 1 → 64 → 64 → 1, with ReLU activations so it can learn curves and not only straight lines.
5. **Train:** mean squared error (MSE) loss, Adam optimizer (learning rate 0.01), batch size 32, 200 epochs. Training and validation loss are recorded every epoch.
6. **Evaluate:** predictions are converted back to the original scale and scored on the test set with MSE, MAE and R². An R² close to 1 means the predictions explain almost all the variation in the real values.
7. **Predict unseen inputs:** six new inputs are compared with the true values from the formula.

## Results
- `loss_curve.png`: training and validation loss over the epochs. Both fall quickly and then flatten, and staying close together means the model generalises.
- `actual_vs_predicted.png`: every test point against its prediction (the closer to the diagonal, the better), and the curve the model learned drawn over the data.
- The exact test MSE, MAE and R² are printed in Step 8 of the notebook, and the explanation of the results is in Step 11.

**Limitation:** the model only saw inputs between -5 and 5, so its predictions outside that range are unreliable. A network learns the pattern in its training data, not the underlying formula.

## Extra project: Digit Recognizer (CNN + drawing app)
An additional project in this folder that recognises handwritten digits from the MNIST dataset.

- **Model (`model.py`):** a convolutional neural network with two blocks (32 and 64 filters, batch normalisation, ReLU, max pooling), then dropout (0.3) and fully connected layers 3136 → 128 → 10.
- **Training (`train.py`):** 54,000 training and 6,000 validation images, Adam (learning rate 0.001), batch size 128, 10 epochs. Random rotations, shifts and zooms are added to the training images so the model copes with messy hand-drawn digits.
- **Result (`evaluate.py`):** **99.30% test accuracy** on the 10,000 test images. Plots are in `images/` (`training_curves.png`, `confusion_matrix.png`, `misclassified.png`).
- **Drawing app (`app.py` and `index.html`):** a small Flask web page where you draw a digit with the mouse and the model predicts it live.

## Files
- `numerical_pattern_predictor.ipynb`: code, outputs and written explanation
- `loss_curve.png`, `actual_vs_predicted.png`: graphs from the notebook
- `model.py`, `preprocess.py`, `train.py`, `evaluate.py`, `app.py`, `index.html`: digit recognizer code
- `digit_cnn.pth`, `history.json`, `results.json`: saved weights, training history and test results
- `images/`: digit recognizer plots

## How to run
**Notebook:** install the libraries with `pip install torch numpy matplotlib`, open the notebook in VS Code, select a Python kernel and click **Run All**.

**Digit recognizer:** install `pip install torch torchvision numpy matplotlib flask pillow`, then run these in this folder:
- `python train.py` to train the model (creates `digit_cnn.pth`)
- `python evaluate.py` to print the test accuracy and save the plots
- `python app.py` and open http://127.0.0.1:5000 to draw digits