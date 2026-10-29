# SVM Binary Classification with Performance Comparison

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data[:, :2]
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

kernels = ['linear', 'poly', 'rbf']
accuracy, precision, f1 = [], [], []

# Train and evaluate SVM models
for k in kernels:
    model = SVC(kernel=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))

# Display results
for i in range(3):
    print(kernels[i], "Accuracy:", round(accuracy[i], 3),
          "Precision:", round(precision[i], 3),
          "F1 Score:", round(f1[i], 3))

# Plot bar chart
x = np.arange(3)
w = 0.25

plt.bar(x-w, accuracy, w, label='Accuracy')
plt.bar(x, precision, w, label='Precision')
plt.bar(x+w, f1, w, label='F1 Score')

plt.xticks(x, kernels)
plt.xlabel("SVM Kernels")
plt.ylabel("Score")
plt.title("SVM Performance Comparison")
plt.legend()
plt.show()
