"""
predict.py

Usage:
    python predict.py image.jpg

Prints exactly one float to stdout: P(image is a photo of a screen),
0.0 = definitely real, 1.0 = definitely a screen.
No other output goes to stdout. Errors are reported on stderr with a
non-zero exit code so the script fails loudly rather than printing
garbage to stdout.
"""

import sys

import config
from utils import load_image, preprocess_image, logger
from features import extract_all_features, features_to_vector
from classifier import load_model, predict_proba_screen


def predict(image_path: str) -> float:
    model, scaler, _ = load_model()
    img = load_image(image_path)
    img = preprocess_image(img)
    feats = extract_all_features(img)
    vec = features_to_vector(feats).reshape(1, -1)
    proba = predict_proba_screen(model, scaler, vec)
    return float(proba[0])


def main():
    if len(sys.argv) != 2:
        logger.error("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        score = predict(image_path)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

    print(f"{score:.{config.PREDICTION_DECIMALS}f}")


if __name__ == "__main__":
    main()
