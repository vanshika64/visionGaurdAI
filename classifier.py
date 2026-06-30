"""
Model construction, training, prediction, and persistence
using only a Random Forest classifier.
"""

import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import config


def build_model() -> RandomForestClassifier:
    """Create a Random Forest classifier."""
    return RandomForestClassifier(**config.RF_PARAMS)


def train_model(X: np.ndarray, y: np.ndarray):
    """
    Train the Random Forest model.

    Returns:
        Trained RandomForestClassifier
    """
    model = build_model()
    model.fit(X, y)
    return model


def save_model(model, feature_names: list, path: str = None):
    """
    Save the trained model and feature names.
    """
    path = path or config.MODEL_PATH

    joblib.dump(model, path)

    with open(config.FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)


def load_model(path: str = None):
    """
    Load the trained Random Forest model.
    """
    path = path or config.MODEL_PATH
    return joblib.load(path)


def load_feature_names(path: str = None) -> list:
    """
    Load saved feature names.
    """
    path = path or config.FEATURE_NAMES_PATH

    with open(path) as f:
        return json.load(f)


def predict_proba_screen(model, X: np.ndarray) -> np.ndarray:
    """
    Returns the probability that each image belongs
    to the 'screen' class.

    Label convention:
        0 -> real
        1 -> screen
    """
    probabilities = model.predict_proba(X)

    screen_index = list(model.classes_).index(1)

    return probabilities[:, screen_index]