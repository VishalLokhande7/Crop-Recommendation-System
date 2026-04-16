import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/crop_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def evaluate_model(name, model, X_test, y_test, X_train, y_train):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} Evaluation ===")
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
    print(f"5-fold CV accuracy mean: {cv_scores.mean():.4f}")
    return cv_scores.mean(), acc


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = {
        "max_depth": [3, 5, 10],
        "criterion": ["gini", "entropy"],
    }
    dt_search = GridSearchCV(dt, dt_grid, cv=5, n_jobs=-1, scoring="accuracy")
    dt_search.fit(X_train, y_train)
    best_dt = dt_search.best_estimator_
    dt_cv_mean, dt_acc = evaluate_model("Decision Tree", best_dt, X_test, y_test, X_train, y_train)
    results.append({
        "name": "Decision Tree",
        "model": best_dt,
        "cv_score": dt_cv_mean,
        "test_accuracy": dt_acc,
        "params": dt_search.best_params_,
    })

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_cv_mean, nb_acc = evaluate_model("Gaussian Naive Bayes", nb, X_test, y_test, X_train, y_train)
    results.append({
        "name": "GaussianNB",
        "model": nb,
        "cv_score": nb_cv_mean,
        "test_accuracy": nb_acc,
        "params": nb.get_params(),
    })

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
    }
    rf_search = GridSearchCV(rf, rf_grid, cv=5, n_jobs=-1, scoring="accuracy")
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    rf_cv_mean, rf_acc = evaluate_model("Random Forest", best_rf, X_test, y_test, X_train, y_train)
    results.append({
        "name": "Random Forest",
        "model": best_rf,
        "cv_score": rf_cv_mean,
        "test_accuracy": rf_acc,
        "params": rf_search.best_params_,
    })

    # SVM with scaler pipeline
    svm_pipe = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("svc", SVC()),
        ]
    )
    svm_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["rbf", "poly"],
    }
    svm_search = GridSearchCV(svm_pipe, svm_grid, cv=5, n_jobs=-1, scoring="accuracy")
    svm_search.fit(X_train, y_train)
    best_svm = svm_search.best_estimator_
    svm_cv_mean, svm_acc = evaluate_model("SVM", best_svm, X_test, y_test, X_train, y_train)
    results.append({
        "name": "SVM",
        "model": best_svm,
        "cv_score": svm_cv_mean,
        "test_accuracy": svm_acc,
        "params": svm_search.best_params_,
    })

    best_entry = max(results, key=lambda item: item["cv_score"])
    best_model = best_entry["model"]
    best_name = best_entry["name"]
    best_params = best_entry["params"]
    best_test_acc = best_entry["test_accuracy"]
    best_cv = best_entry["cv_score"]

    best_model_path = MODEL_DIR / "best_model.pkl"
    with open(best_model_path, "wb") as model_file:
        pickle.dump(best_model, model_file)

    print("\n=== Best Model ===")
    print(f"Name: {best_name}")
    print(f"CV accuracy: {best_cv:.4f}")
    print(f"Test accuracy: {best_test_acc:.4f}")
    print(f"Parameters: {best_params}")
    print(f"Saved best model to {best_model_path}")


if __name__ == "__main__":
    main()
