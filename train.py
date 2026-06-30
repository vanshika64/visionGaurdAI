"""
train.py

Loads the dataset, extracts handcrafted features, trains a Random Forest
classifier, evaluates it, performs cross-validation, and saves the trained
model.

Usage:
    python train.py
"""

import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)

import config
from utils import logger, list_image_files, load_image, preprocess_image
from features import (
    extract_all_features,
    features_to_vector,
    get_feature_names,
)
from classifier import (
    train_model,
    save_model,
    predict_proba_screen,
)
from evaluate import compute_metrics, print_report

LABEL_REAL = 0
LABEL_SCREEN = 1


def build_dataset():
    """
    Load all images, extract features, and build X and y.
    """
    feature_names = get_feature_names()

    X_rows = []
    y_rows = []

    for label, directory, label_name in [
        (LABEL_REAL, config.REAL_DIR, "real"),
        (LABEL_SCREEN, config.SCREEN_DIR, "screen"),
    ]:
        files = list_image_files(directory)

        if not files:
            logger.warning(f"No images found in {directory}")

        for path in files:
            try:
                img = load_image(path)
                img = preprocess_image(img)

                features = extract_all_features(img)
                vector = features_to_vector(features)

                X_rows.append(vector)
                y_rows.append(label)

            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")

        logger.info(f"Loaded {len(files)} candidate images from '{label_name}'")

    if not X_rows:
        raise RuntimeError(
            "No usable images found. Check dataset folders."
        )

    X = np.vstack(X_rows)
    y = np.array(y_rows)

    return X, y, feature_names


def main():

    logger.info("Building dataset (extracting features for all images)...")

    X, y, feature_names = build_dataset()

    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(
        f"Class balance: real={np.sum(y==LABEL_REAL)}, "
        f"screen={np.sum(y==LABEL_SCREEN)}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    # Train Random Forest
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = predict_proba_screen(model, X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)

    print_report(metrics, title="Held-out Test Set")

    # Cross-validation
    cv_model = train_model(X, y)

    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )

    cv_scores = cross_val_score(
        cv_model,
        X,
        y,
        cv=skf,
        scoring="accuracy",
    )

    logger.info(
        f"{config.CV_FOLDS}-fold CV Accuracy: "
        f"mean={cv_scores.mean():.4f} "
        f"std={cv_scores.std():.4f}"
    )

    # Retrain using the full dataset
    final_model = train_model(X, y)

    save_model(final_model, feature_names)

    logger.info(f"Saved model to {config.MODEL_PATH}")

    # Feature Importance
    importances = final_model.feature_importances_

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: -x[1]
    )

    logger.info("Top 10 Feature Importances:")

    for name, score in ranked[:10]:
        logger.info(f"  {name}: {score:.4f}")


if __name__ == "__main__":
    main()