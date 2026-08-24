import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/sara3691/ml/main/student_dataset.csv")

num = df.select_dtypes(include='number')

print("Mean:")
print(num.mean())

print("\nMedian:")
print(num.median())

print("\nMode:")
print(num.mode().iloc[0])

print("\nVariance:")
print(num.var())

print("\nStandard Deviation:")
print(num.std())
