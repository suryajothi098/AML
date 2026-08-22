import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Iris.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
df.info()

df["Species"] = LabelEncoder().fit_transform(df["Species"])

X = df.iloc[:, 1:-1]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
kp = knn.predict(X_test)

print("\nKNN Accuracy:", accuracy_score(y_test, kp))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, kp))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
np = nb.predict(X_test)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, np))
print("Naive Bayes Confusion Matrix:\n", confusion_matrix(y_test, np))

# Plot
plt.plot(y_test.values, 'o-', label="Actual")
plt.plot(kp, '*-', label="KNN")
plt.plot(np, '^-', label="Naive Bayes")
plt.title("KNN and Naive Bayes Classification")
plt.xlabel("Test Samples")
plt.ylabel("Species")
plt.legend()
plt.grid()
plt.show()
