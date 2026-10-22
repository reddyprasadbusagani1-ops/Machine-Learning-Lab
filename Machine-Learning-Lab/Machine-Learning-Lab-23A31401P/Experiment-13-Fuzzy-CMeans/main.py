# Aim: Demonstrate Fuzzy C-Means clustering.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz

X, _ = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)
# scikit-fuzzy expects features x samples
data = X_scaled.T
centers, membership, initial_membership, distances, objective, iterations, fpc = fuzz.cluster.cmeans(
    data, c=3, m=2.0, error=0.005, maxiter=1000, init=None, seed=42
)
labels = np.argmax(membership, axis=0)
print("Fuzzy C-Means Clustering (Iris)")
print("Number of clusters:", centers.shape[0])
print("Iterations:", iterations)
print("Fuzzy partition coefficient:", round(float(fpc), 4))
print("First 10 hard labels derived from membership:", labels[:10].tolist())
print("First 10 membership vectors:\n", np.round(membership[:, :10].T, 3))

plt.scatter(data[0], data[1], c=labels, s=35)
plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=180, edgecolors="black")
plt.xlabel("Standardized feature 1")
plt.ylabel("Standardized feature 2")
plt.title("Fuzzy C-Means (first two features)")
plt.tight_layout()
plt.savefig("Experiment-13-Fuzzy-CMeans/plot.png", dpi=150)
plt.show()
