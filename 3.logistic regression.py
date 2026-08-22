import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/sara3691/ml/main/student_dataset.csv")

# Convert Result to 0 and 1
if df["Result"].dtype == "object":
    df["Result"] = df["Result"].map({"Fail": 0, "Pass": 1})

# Input and output
X = df[["Study_Hours", "Attendance", "Internal_Mark"]]
y = df["Result"]

# Check classes
print("Result counts:")
print(y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot
plt.scatter(df["Study_Hours"], df["Result"])
plt.xlabel("Study Hours")
plt.ylabel("Result")
plt.title("Logistic Regression - Pass/Fail")
plt.yticks([0, 1], ["Fail", "Pass"])
plt.show()
