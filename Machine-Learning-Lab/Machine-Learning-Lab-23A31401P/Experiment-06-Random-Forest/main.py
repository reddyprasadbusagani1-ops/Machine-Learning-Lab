# Aim: Apply Random Forest for both classification and regression.
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

Xc, yc = load_iris(return_X_y=True)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.25, random_state=42, stratify=yc
)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(Xc_train, yc_train)
pc = clf.predict(Xc_test)
print("Random Forest Classification (Iris)")
print("Accuracy:", round(accuracy_score(yc_test, pc), 4))

Xr, yr = load_diabetes(return_X_y=True)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.25, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(Xr_train, yr_train)
pr = reg.predict(Xr_test)
print("\nRandom Forest Regression (Diabetes)")
print("MSE:", round(mean_squared_error(yr_test, pr), 4))
print("R2 score:", round(r2_score(yr_test, pr), 4))
