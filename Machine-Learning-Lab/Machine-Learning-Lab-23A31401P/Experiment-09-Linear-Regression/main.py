# Aim: Demonstrate simple linear regression for a regression problem.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Reproducible illustrative data: target approximately increases with study hours.
rng = np.random.default_rng(42)
hours = np.arange(1, 31, dtype=float).reshape(-1, 1)
scores = 35 + 2.1 * hours.ravel() + rng.normal(0, 4, size=len(hours))
X_train, X_test, y_train, y_test = train_test_split(hours, scores, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Simple Linear Regression")
print("Intercept:", round(model.intercept_, 3))
print("Slope:", round(model.coef_[0], 3))
print("MSE:", round(mean_squared_error(y_test, pred), 3))
print("R2 score:", round(r2_score(y_test, pred), 3))

plt.scatter(hours, scores, label="Observed data")
line_x = np.linspace(hours.min(), hours.max(), 100).reshape(-1, 1)
plt.plot(line_x, model.predict(line_x), label="Regression line")
plt.xlabel("Study hours")
plt.ylabel("Illustrative score")
plt.title("Simple Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("Experiment-09-Linear-Regression/plot.png", dpi=150)
plt.show()
