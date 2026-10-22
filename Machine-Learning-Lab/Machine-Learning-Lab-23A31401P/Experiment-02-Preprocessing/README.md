# Experiment 2: Data Pre-processing

## Aim
To demonstrate attribute selection, handling missing values, discretization, and outlier elimination.

## Theory
Pre-processing prepares raw data for machine learning. Attribute selection keeps relevant columns; imputation fills missing values; discretization converts continuous values to bins; the IQR rule identifies extreme values.

## Algorithm
1. Create a dataset with missing values and a possible outlier.\n2. Select relevant attributes.\n3. Fill missing numerical values with medians.\n4. Convert age into bins.\n5. Use the IQR rule to filter outliers.

## Program
Run `main.py` from the repository root:

```cmd
python Experiment-02-Preprocessing/main.py
```

## Requirements
Install packages from the root `requirements.txt` first.

## Result
The program prints relevant statistics or evaluation metrics. For experiments that create plots, a plot window is displayed and the plot is saved as `plot.png` in the experiment folder.

