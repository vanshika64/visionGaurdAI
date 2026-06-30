"""
Central configuration for the screen-vs-real classifier.
No logic here -- only constants used across features.py, classifier.py,
train.py, predict.py and evaluate.py.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
REAL_DIR = os.path.join(DATASET_DIR, "real")
SCREEN_DIR = os.path.join(DATASET_DIR, "screen")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

RESIZE_DIM = 512

FFT_BANDS = [
    (0.0, 0.15),
    (0.15, 0.40),
    (0.40, 0.70),
    (0.70, 1.00),
]

MOIRE_LOW_FREQ_RADIUS = 0.05
MOIRE_BAND_RADIUS = (0.10, 0.45)
MOIRE_PEAK_PERCENTILE = 99.0

WAVELET_NAME = "db4"
WAVELET_LEVELS = 2

LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

CANNY_LOW = 50
CANNY_HIGH = 150

HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 60
HOUGH_MAX_LINE_GAP = 10
AXIS_ALIGN_TOLERANCE_DEG = 5.0

REFLECTION_BRIGHTNESS_THRESH = 235
REFLECTION_MIN_AREA = 15
REFLECTION_MAX_AREA_FRACTION = 0.05

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Random Forest parameters
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Deployment model
DEFAULT_MODEL_CHOICE = "rf"

# Output formatting
PREDICTION_DECIMALS = 6