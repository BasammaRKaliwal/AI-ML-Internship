# Assignment 2: Neural Network Comparison

**Notebook:** [assignment2_neural_network_comparison.ipynb](assignment2_neural_network_comparison.ipynb)

## Goal
Compare a linear model with a nonlinear neural network on a synthetic two-class dataset, and test how the hidden-layer size and the learning rate change the result.

## Data
1,000 points shaped like two interleaving half-circles (noise 0.25), split into 700 train, 150 validation and 150 test samples. Inputs are normalised using training statistics only.

## Models
- **Linear model:** one layer (2 → 1). It can only draw a straight decision line.
- **Neural network:** 2 → hidden → hidden → 1 with ReLU activations, so it can draw a curved boundary.

## Experiments
- Hidden sizes: 2, 4, 8, 16, 32, 64
- Learning rates: 0.001, 0.01, 0.05, 0.1
- Adam optimizer, binary cross-entropy loss, 300 full-batch epochs
- The best setting is chosen by **validation** accuracy, then its **test** accuracy is reported, so the test set is never used for tuning.

## Results
| Model | Test accuracy |
|---|---|
| Linear model | 90.0% |
| Best neural network (16 hidden units, learning rate 0.01) | 94.7% |

- Weakest setting: 2 hidden units with learning rate 0.001 (62.7% validation accuracy). The network was too small and learned too slowly.
- Largest train/validation gap: 32 hidden units with learning rate 0.1 (3.6 points), a sign of overfitting.

## Conclusion
The neural network performed best because the two classes are curved and a straight line cannot separate them fully. I would pick 16 hidden units with a learning rate of 0.01, since it gave high accuracy without a large train/validation gap.

## Files
- `assignment2_neural_network_comparison.ipynb`: code, tables and written conclusion
- `decision_boundaries.png`: linear model vs neural network boundaries
- `hyperparameter_heatmap.png`: accuracy for every hidden size and learning rate
- `confusion_matrices.png`: confusion matrices on the test set

## How to run
Install the libraries with `pip install torch numpy matplotlib`, open the notebook in VS Code, select a Python kernel and click **Run All**. It takes about a minute or two.