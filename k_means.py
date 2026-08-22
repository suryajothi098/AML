import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("https://raw.githubusercontent.com/sara3691/ml/main/student_dataset.csv")

# Select features
X = data[["Study_Hours", "Attendance"]]

# Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Plot clusters
plt.scatter(X["Study_Hours"], X["Attendance"], c=y_kmeans)

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance")
plt.title("K-Means Clustering")
plt.show()
