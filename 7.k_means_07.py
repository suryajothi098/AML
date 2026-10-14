
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load built-in dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Select features
X = X[["sepal length (cm)", "sepal width (cm)"]]

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Plot clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1],
            c=y_kmeans, cmap="viridis", s=50)

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            color="red", marker="X", s=200,
            label="Centroids")

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("K-Means Clustering - Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()
