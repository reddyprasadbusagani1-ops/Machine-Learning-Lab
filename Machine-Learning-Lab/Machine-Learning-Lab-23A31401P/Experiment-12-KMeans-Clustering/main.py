# Aim: Implement K-means clustering and evaluate different values of K.
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

X, _ = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)
scores = []
ks = range(2, 8)
print("K-means clustering: sum of Euclidean distances to nearest center")
for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    distances = pairwise_distances(X_scaled, model.cluster_centers_)
    total_distance = distances[range(len(labels)), labels].sum()
    scores.append(total_distance)
    print(f"K={k}: sum distance={total_distance:.4f}, inertia={model.inertia_:.4f}")

plt.plot(list(ks), scores, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Sum of Euclidean distances to assigned center")
plt.title("K-means Performance vs K")
plt.tight_layout()
plt.savefig("Experiment-12-KMeans-Clustering/plot.png", dpi=150)
plt.show()
