import pandas as pd

# Load the dataset
df = pd.read_csv("‪Iris.csv")

# Select only numerical columns
num = df.select_dtypes(include='number')

# Compute statistical measures
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
