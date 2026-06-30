"""
Model construction, training and persistence.
Keeps train.py free of model-internals so the same factories can be reused
by evaluate.py or future experiments.
"""

import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import config


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(**config.RF_PARAMS)


def build_svm() -> SVC:
    return SVC(**config.SVM_PARAMS)


def build_model(model_choice: str):
    if model_choice == "rf":
        return build_random_forest(), False  # False = no scaler needed
    elif model_choice == "svm":
        return build_svm(), True  # True = needs a fitted scaler
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")


def train_model(X: np.ndarray, y: np.ndarray, model_choice: str):
    """
    Fits the requested model. Returns (fitted_model, fitted_scaler_or_None).
    """
    model, needs_scaler = build_model(model_choice)
    scaler = None
    X_input = X
    if needs_scaler:
        scaler = StandardScaler()
        X_input = scaler.fit_transform(X)
    model.fit(X_input, y)
    return model, scaler


def save_model(model, scaler, feature_names: list, model_choice: str, path: str = None):
    path = path or config.MODEL_PATH
    bundle = {
        "model": model,
        "scaler": scaler,
        "model_choice": model_choice,
    }
    joblib.dump(bundle, path)

    with open(config.FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)


def load_model(path: str = None):
    path = path or config.MODEL_PATH
    bundle = joblib.load(path)
    return bundle["model"], bundle["scaler"], bundle["model_choice"]


def load_feature_names(path: str = None) -> list:
    path = path or config.FEATURE_NAMES_PATH
    with open(path) as f:
        return json.load(f)


def predict_proba_screen(model, scaler, X: np.ndarray) -> np.ndarray:
    """
    Returns P(class == screen) for each row in X.
    Assumes label convention 1 = screen, 0 = real (enforced in train.py).
    """
    X_input = scaler.transform(X) if scaler is not None else X
    proba = model.predict_proba(X_input)
    # locate the column for class label 1
    classes = list(model.classes_)
    screen_idx = classes.index(1)
    return proba[:, screen_idx]
