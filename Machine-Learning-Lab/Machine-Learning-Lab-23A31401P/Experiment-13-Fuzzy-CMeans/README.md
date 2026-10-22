# Experiment 13: Fuzzy C-Means Clustering

## Aim
To demonstrate fuzzy C-means clustering on a selected dataset.

## Theory
Unlike hard clustering, Fuzzy C-Means assigns each observation a membership degree for every cluster. The membership degrees are between 0 and 1 and sum to 1 for each observation. The fuzzifier m controls how soft the assignments are.

## Algorithm
1. Load and standardize the Iris features.\n2. Run fuzzy c-means with three clusters and fuzzifier m=2.\n3. Display membership values and the fuzzy partition coefficient.\n4. Plot a two-dimensional view using the first two features.

## Program
Run `main.py` from the repository root:

```cmd
python Experiment-13-Fuzzy-CMeans/main.py
```

## Requirements
Install packages from the root `requirements.txt` first.

## Result
The program prints relevant statistics or evaluation metrics. For experiments that create plots, a plot window is displayed and the plot is saved as `plot.png` in the experiment folder.

