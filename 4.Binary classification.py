import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to binary classification
y = np.where(y == 0, 0, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

# Train SVM with different kernels
for k in kernels:
    svm = SVC(kernel=k)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics["Accuracy"].append(acc)
    metrics["Precision"].append(prec)
    metrics["Recall"].append(rec)
    metrics["F1 Score"].append(f1)

    print("\nKernel:", k)
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f1, 4))

# Plot graph
x = np.arange(len(kernels))
width = 0.2

plt.bar(x - 0.3, metrics["Accuracy"], width, label="Accuracy")
plt.bar(x - 0.1, metrics["Precision"], width, label="Precision")
plt.bar(x + 0.1, metrics["Recall"], width, label="Recall")
plt.bar(x + 0.3, metrics["F1 Score"], width, label="F1 Score")

plt.xticks(x, kernels)
plt.xlabel("Kernel")
plt.ylabel("Score")
plt.title("SVM Performance with Different Kernels")
plt.legend()
plt.show()
