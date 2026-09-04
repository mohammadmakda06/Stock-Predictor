"""
train.py
--------
Trains and evaluates Random Forest and Gradient Boosting classifiers to
predict short-term stock trend direction.

- Chronological (non-shuffled) train/test split, since stock data is a time
  series and shuffling would leak future information into training.
- Hyperparameter tuning via GridSearchCV with TimeSeriesSplit cross-validation.
- Evaluation: accuracy, precision, recall, F1, confusion matrix, and
  feature importances.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


def chronological_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def tune_random_forest(X_train, y_train, n_splits: int = 3):
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [4, 8],
        "min_samples_leaf": [5, 20],
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=1),
        param_grid, cv=tscv, scoring="f1", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_gradient_boosting(X_train, y_train, n_splits: int = 3):
    param_grid = {
        "n_estimators": [100, 150],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid, cv=tscv, scoring="f1", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    return metrics, cm, report


def top_feature_importances(model, feature_names, top_n: int = 12) -> pd.Series:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n)
