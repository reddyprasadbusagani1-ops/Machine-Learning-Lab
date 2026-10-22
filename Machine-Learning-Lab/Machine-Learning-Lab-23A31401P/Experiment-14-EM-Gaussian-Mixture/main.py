# Aim: Demonstrate Expectation-Maximization based clustering using Gaussian Mixture Models.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

X, _ = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)
model = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=5)
model.fit(X_scaled)
labels = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)
print("EM-based Gaussian Mixture clustering (Iris)")
print("Converged:", model.converged_)
print("Iterations:", model.n_iter_)
print("AIC:", round(model.aic(X_scaled), 3))
print("BIC:", round(model.bic(X_scaled), 3))
print("Silhouette score:", round(silhouette_score(X_scaled, labels), 4))
print("First 5 membership probabilities:\n", probabilities[:5].round(3))

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, s=35)
plt.xlabel("Standardized feature 1")
plt.ylabel("Standardized feature 2")
plt.title("Gaussian Mixture Model Clustering")
plt.tight_layout()
plt.savefig("Experiment-14-EM-Gaussian-Mixture/plot.png", dpi=150)
plt.show()
