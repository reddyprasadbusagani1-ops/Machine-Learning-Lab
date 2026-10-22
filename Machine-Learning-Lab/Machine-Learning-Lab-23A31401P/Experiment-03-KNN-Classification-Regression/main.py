# Aim: Apply K-nearest neighbors to classification and regression tasks.
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Classification: Iris dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
classifier = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
print("KNN Classification (Iris)")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))

# Regression: California Housing (fetched by sklearn; needs internet first time)
try:
    Xr, yr = fetch_california_housing(return_X_y=True)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr, yr, test_size=0.25, random_state=42
    )
    # Use a manageable subset for a fast beginner lab run
    Xr_train, yr_train = Xr_train[:3000], yr_train[:3000]
    Xr_test, yr_test = Xr_test[:1000], yr_test[:1000]
    regressor = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    regressor.fit(Xr_train, yr_train)
    reg_pred = regressor.predict(Xr_test)
    print("\nKNN Regression (California Housing)")
    print("MSE:", round(mean_squared_error(yr_test, reg_pred), 4))
    print("R2 score:", round(r2_score(yr_test, reg_pred), 4))
except Exception as exc:
    print("\nKNN regression dataset could not be loaded:", exc)
    print("Try again with an internet connection; classification completed.")
