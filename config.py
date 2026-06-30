"""
Central configuration for the screen-vs-real classifier.
No logic here -- only constants used across features.py, classifier.py,
train.py, predict.py and evaluate.py.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
REAL_DIR = os.path.join(DATASET_DIR, "real")
SCREEN_DIR = os.path.join(DATASET_DIR, "screen")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
# All images are resized to this square size before any feature extraction.
# Fixed size is mandatory because FFT/wavelet/LBP features are resolution
# sensitive -- inconsistent sizing between train and predict would silently
# corrupt accuracy.
RESIZE_DIM = 512

# ---------------------------------------------------------------------------
# FFT features
# ---------------------------------------------------------------------------
# Radial frequency bands as fractions of the Nyquist radius (0.0 - 1.0)
FFT_BANDS = [(0.0, 0.15), (0.15, 0.4), (0.4, 0.7), (0.7, 1.0)]

# ---------------------------------------------------------------------------
# Moire detection
# ---------------------------------------------------------------------------
MOIRE_LOW_FREQ_RADIUS = 0.05   # exclude DC / very-low-freq region
MOIRE_BAND_RADIUS = (0.1, 0.45)  # band searched for paired peaks
MOIRE_PEAK_PERCENTILE = 99.0     # percentile threshold for peak detection

# ---------------------------------------------------------------------------
# Wavelet features
# ---------------------------------------------------------------------------
WAVELET_NAME = "db4"
WAVELET_LEVELS = 2

# ---------------------------------------------------------------------------
# LBP features
# ---------------------------------------------------------------------------
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

# ---------------------------------------------------------------------------
# Edge / Hough features
# ---------------------------------------------------------------------------
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 60
HOUGH_MAX_LINE_GAP = 10
AXIS_ALIGN_TOLERANCE_DEG = 5.0  # degrees from 0/90 considered "axis aligned"

# ---------------------------------------------------------------------------
# Reflection detection
# ---------------------------------------------------------------------------
REFLECTION_BRIGHTNESS_THRESH = 235  # 0-255 scale
REFLECTION_MIN_AREA = 15            # pixels, ignore tiny noise blobs
REFLECTION_MAX_AREA_FRACTION = 0.05  # ignore overexposed whole-image blobs

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

SVM_PARAMS = dict(
    kernel="rbf",
    C=2.0,
    gamma="scale",
    probability=True,
    random_state=RANDOM_SEED,
)

# Which model train.py should save as the final model.pkl: "rf" or "svm".
# train.py evaluates both and prints results; this is the fallback default
# if you don't override the choice on the command line.
DEFAULT_MODEL_CHOICE = "rf"

# Output formatting for predict.py
PREDICTION_DECIMALS = 6
