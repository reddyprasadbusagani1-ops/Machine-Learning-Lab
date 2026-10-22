# Experiment 3: KNN Classification and Regression

## Aim
To apply K-nearest neighbors for a classification task and a regression task.

## Theory
KNN predicts using the nearest training examples. Classification predicts a class label; regression predicts a numeric value. Feature scaling is important because KNN uses distances.

## Algorithm
1. Load a classification dataset and split into train/test sets.\n2. Scale features and fit KNN classifier.\n3. Evaluate accuracy.\n4. Load a regression dataset, scale features, fit KNN regressor, and report MSE and R².

## Program
Run `main.py` from the repository root:

```cmd
python Experiment-03-KNN-Classification-Regression/main.py
```

## Requirements
Install packages from the root `requirements.txt` first.

## Result
The program prints relevant statistics or evaluation metrics. For experiments that create plots, a plot window is displayed and the plot is saved as `plot.png` in the experiment folder.
\n**Dataset note:** The Iris dataset is bundled with scikit-learn. California Housing is fetched on first use and may require an internet connection.
