# Experiment 14: Expectation-Maximization Based Clustering

## Aim
To demonstrate clustering using a Gaussian Mixture Model fitted with the Expectation-Maximization algorithm.

## Theory
A Gaussian Mixture Model assumes data are generated from a mixture of Gaussian distributions. EM alternates between estimating cluster membership probabilities (E-step) and updating model parameters (M-step).

## Algorithm
1. Load and standardize Iris features.\n2. Fit a three-component Gaussian Mixture Model.\n3. Predict cluster assignments and membership probabilities.\n4. Report convergence, AIC, BIC, and silhouette score.\n5. Plot the first two standardized features.

## Program
Run `main.py` from the repository root:

```cmd
python Experiment-14-EM-Gaussian-Mixture/main.py
```

## Requirements
Install packages from the root `requirements.txt` first.

## Result
The program prints relevant statistics or evaluation metrics. For experiments that create plots, a plot window is displayed and the plot is saved as `plot.png` in the experiment folder.

