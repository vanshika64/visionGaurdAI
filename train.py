"""
train.py

Loads the dataset, extracts features for every image, trains both
candidate models (Random Forest and RBF SVM), evaluates each via a
held-out test split and k-fold cross-validation, prints full metrics for
both, and saves the better-performing one as models/model.pkl.

Usage:
    python train.py
    python train.py --model rf      # force a specific model to be saved
    python train.py --model svm
"""

import argparse
import sys

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

import config
from utils import logger, list_image_files, load_image, preprocess_image
from features import extract_all_features, features_to_vector, get_feature_names
from classifier import train_model, save_model, predict_proba_screen
from evaluate import compute_metrics, print_report

LABEL_REAL = 0
LABEL_SCREEN = 1


def build_dataset():
    """
    Walks dataset/real and dataset/screen, extracts features for every
    readable image, and returns (X, y, feature_names).
    Unreadable/corrupt files are skipped with a warning rather than
    crashing the whole run.
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
                feats = extract_all_features(img)
                vec = features_to_vector(feats)
                if vec.shape[0] != len(feature_names):
                    logger.warning(
                        f"Feature length mismatch for {path}, skipping."
                    )
                    continue
                X_rows.append(vec)
                y_rows.append(label)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")
        logger.info(f"Loaded {len(files)} candidate images from '{label_name}'")

    if not X_rows:
        raise RuntimeError(
            "No usable images found. Check dataset/real and dataset/screen."
        )

    X = np.vstack(X_rows)
    y = np.array(y_rows)
    return X, y, feature_names


def evaluate_model_choice(model_choice: str, X_train, X_test, y_train, y_test, X_full, y_full):
    model, scaler = train_model(X_train, y_train, model_choice)

    X_test_input = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_test_input)
    y_proba = predict_proba_screen(model, scaler, X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_report(metrics, title=f"Held-out Test Set ({model_choice})")

    # Cross-validation on the full dataset for a more stable accuracy
    # estimate given how small the dataset is.
    cv_model, cv_scaler = train_model(X_full, y_full, model_choice)
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    X_cv_input = cv_scaler.transform(X_full) if cv_scaler is not None else X_full
    cv_scores = cross_val_score(cv_model, X_cv_input, y_full, cv=skf, scoring="accuracy")
    logger.info(
        f"{model_choice} {config.CV_FOLDS}-fold CV accuracy: "
        f"mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}"
    )

    return model, scaler, metrics, cv_scores.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["rf", "svm", "auto"],
        default="auto",
        help="Which model to save as the final model.pkl. "
        "'auto' picks whichever scores higher on held-out test accuracy.",
    )
    args = parser.parse_args()

    logger.info("Building dataset (extracting features for all images)...")
    X, y, feature_names = build_dataset()
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Class balance: real={np.sum(y == LABEL_REAL)}, screen={np.sum(y == LABEL_SCREEN)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    results = {}
    for model_choice in ["rf", "svm"]:
        model, scaler, metrics, cv_mean = evaluate_model_choice(
            model_choice, X_train, X_test, y_train, y_test, X, y
        )
        results[model_choice] = {
            "model": model,
            "scaler": scaler,
            "metrics": metrics,
            "cv_mean": cv_mean,
        }

    if args.model == "auto":
        chosen = max(results, key=lambda k: results[k]["cv_mean"])
    else:
        chosen = args.model

    logger.info(f"Selected model for deployment: '{chosen}'")

    # Retrain chosen model on the FULL dataset (train+test) for the final
    # saved artifact, maximizing data used for the deployed model.
    final_model, final_scaler = train_model(X, y, chosen)
    save_model(final_model, final_scaler, feature_names, chosen)
    logger.info(f"Saved model to {config.MODEL_PATH}")

    # Feature importance (Random Forest only) -- useful for Step 10 review.
    if chosen == "rf":
        importances = final_model.feature_importances_
        ranked = sorted(zip(feature_names, importances), key=lambda t: -t[1])
        logger.info("Top 10 feature importances:")
        for name, imp in ranked[:10]:
            logger.info(f"  {name}: {imp:.4f}")


if __name__ == "__main__":
    main()
