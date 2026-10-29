import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load built-in dataset
data = load_iris()
X, y = data.data, data.target

# Select two classes for binary classification
X, y = X[y != 2], y[y != 2]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# KNN algorithm
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
kp = knn.predict(X_test)

# Naive Bayes algorithm
nb = GaussianNB()
nb.fit(X_train, y_train)
np = nb.predict(X_test)

# Display accuracy
print("KNN Accuracy:", accuracy_score(y_test, kp))
print("Naive Bayes Accuracy:", accuracy_score(y_test, np))

# Plot actual and predicted results
plt.plot(y_test, 'o-', color='blue', label="Actual", linewidth=2)
plt.plot(kp + 0.03, '*-', color='red', label="KNN")
plt.plot(np - 0.03, '^-', color='green', label="Naive Bayes")

plt.title("KNN and Naive Bayes Classification")
plt.xlabel("Test Samples")
plt.ylabel("Result (0 or 1)")
plt.yticks([0, 1])
plt.legend()
plt.grid()
plt.show()
