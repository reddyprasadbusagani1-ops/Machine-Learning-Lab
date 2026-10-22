# Experiment 4: Decision Tree Classification and Parameter Tuning

## Aim
To classify a dataset using a decision tree and tune its parameters for better validation performance.

## Theory
A decision tree splits features into decision rules. Parameters such as maximum depth and minimum split size control complexity and can reduce overfitting. Grid search evaluates parameter combinations using cross-validation.

## Algorithm
1. Load the Breast Cancer Wisconsin dataset.\n2. Split the data into training and test sets.\n3. Train a default decision tree and record test accuracy.\n4. Use GridSearchCV to tune depth, split size, and criterion.\n5. Compare accuracy and print the classification report.

## Program
Run `main.py` from the repository root:

```cmd
python Experiment-04-Decision-Tree-Classification/main.py
```

## Requirements
Install packages from the root `requirements.txt` first.

## Result
The program prints relevant statistics or evaluation metrics. For experiments that create plots, a plot window is displayed and the plot is saved as `plot.png` in the experiment folder.

