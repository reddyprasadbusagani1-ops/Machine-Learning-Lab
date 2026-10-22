# Experiment 12: K-means Clustering

## Aim
To implement K-means, measure the sum of Euclidean distances to assigned cluster centers, and compare results for different K values.

## Theory
K-means alternates between assigning points to the nearest centroid and updating centroids. The number of clusters K is a user-selected parameter. The experiment standardizes Iris features and evaluates K from 2 through 7.

## Algorithm
1. Load and standardize Iris features.\n2. Fit KMeans for each K from 2 to 7.\n3. Calculate each observation's Euclidean distance to its assigned center and sum the distances.\n4. Print the sums and plot them against K.\n\n**Note:** The syllabus specifically asks for the sum of Euclidean distances. The program reports this in addition to K-means inertia, which is the sum of squared distances.

## Program
Run `main.py` from the repository root:

```cmd
python Experiment-12-KMeans-Clustering/main.py
```

## Requirements
Install packages from the root `requirements.txt` first.

## Result
The program prints relevant statistics or evaluation metrics. For experiments that create plots, a plot window is displayed and the plot is saved as `plot.png` in the experiment folder.

