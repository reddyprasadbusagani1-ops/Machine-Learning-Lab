# Aim: Train a decision tree classifier and tune parameters using grid search.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
base = DecisionTreeClassifier(random_state=42)
base.fit(X_train, y_train)
base_pred = base.predict(X_test)
print("Default tree accuracy:", round(accuracy_score(y_test, base_pred), 4))

params = {
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}
search = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring="accuracy")
search.fit(X_train, y_train)
best = search.best_estimator_
pred = best.predict(X_test)
print("Best parameters:", search.best_params_)
print("Tuned tree accuracy:", round(accuracy_score(y_test, pred), 4))
print("\nClassification report:\n", classification_report(y_test, pred, target_names=["malignant", "benign"]))
