import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("https://raw.githubusercontent.com/sara3691/ml/main/student_dataset.csv")

# Linear Regression
X = data[['Study_Hours']]
y = data['Final_Mark']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Linear Regression Prediction:")
print(prediction)

plt.scatter(X_test, y_test)
plt.plot(X_test, prediction)
plt.xlabel("Study Hours")
plt.ylabel("Final Mark")
plt.title("Linear Regression")
plt.show()


# Multiple Regression
X = data[['Study_Hours', 'Attendance', 'Internal_Mark']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Multiple Regression Prediction:")
print(prediction)

plt.scatter(y_test, prediction)
plt.xlabel("Actual Final Mark")
plt.ylabel("Predicted Final Mark")
plt.title("Multiple Regression")
plt.show()
