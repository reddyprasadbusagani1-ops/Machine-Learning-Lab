# Aim: Demonstrate decision tree regression on a numeric prediction problem.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Decision Tree Regression (Diabetes dataset)")
print("MAE:", round(mean_absolute_error(y_test, pred), 4))
print("MSE:", round(mean_squared_error(y_test, pred), 4))
print("R2 score:", round(r2_score(y_test, pred), 4))
print("First five predictions:", pred[:5].round(2).tolist())
