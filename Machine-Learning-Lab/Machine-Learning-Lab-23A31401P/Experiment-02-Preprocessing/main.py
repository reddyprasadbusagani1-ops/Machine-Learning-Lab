# Aim: Demonstrate attribute selection, missing-value handling, discretization, and outlier elimination.
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

data = pd.DataFrame({
    "age": [21, 22, np.nan, 24, 25, 26, 120],
    "hours_studied": [2, 4, 3, np.nan, 6, 7, 2],
    "attendance": [70, 75, 72, 80, 85, 90, 20],
    "result": ["Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Fail"]
})
print("Original dataset:\n", data)

# a) Attribute selection
selected = data[["age", "hours_studied", "result"]]
print("\nA) Attribute selection:\n", selected)

# b) Handle missing numerical values with the median
clean = data.copy()
for column in ["age", "hours_studied", "attendance"]:
    clean[column] = clean[column].fillna(clean[column].median())
print("\nB) Missing values filled with column medians:\n", clean)

# c) Discretize age into three equal-width bins
discretizer = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
clean["age_bin"] = discretizer.fit_transform(clean[["age"]]).astype(int)
print("\nC) Age discretization:\n", clean[["age", "age_bin"]])

# d) Remove outliers in age using the IQR rule
q1, q3 = clean["age"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
without_outliers = clean[clean["age"].between(lower, upper)]
print(f"\nD) IQR age bounds: [{lower:.2f}, {upper:.2f}]")
print("Rows after removing age outliers:\n", without_outliers)
