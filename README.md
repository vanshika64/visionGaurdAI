# Real Photo vs. Screen Photo Classifier

Classical computer vision + lightweight ML solution that determines whether
an input image is a real photo of a real-world object, or a photo of a
screen (phone/laptop/monitor/tablet) displaying an image.

Output: a single float in [0, 1] -- 0.0 = definitely real, 1.0 = definitely
a screen.

## Approach

Rather than training a deep CNN (which would badly overfit on a small
hand-collected dataset and likely learn scene content instead of the true
screen-detection cue), this project uses handcrafted features that target
the actual physical artifact of photographing a display: moire patterns
and frequency-domain anomalies caused by the interaction between a
camera's sensor grid and a display's sub-pixel grid. These are combined
with supporting texture, edge, color, and reflection features and fed into
a Random Forest / RBF SVM classifier (whichever scores better via
cross-validation, selected automatically by `train.py`).

See `features.py` for the full feature set: FFT radial-band energy,
moire peak detection, wavelet sub-band energy, edge density / Hough-line
bezel detection, LBP texture, gradient orientation peakiness, Laplacian
variance, color/saturation statistics, brightness distribution, and
reflection blob detection.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

```
dataset/
    real/      # 50+ real-world photos
    screen/    # 50+ photos of screens displaying images
```

See project notes for data collection guidelines (lighting, angle,
distance, device, and reflection diversity).

## Training

```bash
python train.py
```

Trains and evaluates both Random Forest and RBF SVM via a held-out test
split and 5-fold cross-validation, prints accuracy / precision / recall /
F1 / ROC-AUC / confusion matrix for each, then saves the better-performing
model to `models/model.pkl` (retrained on the full dataset).

Force a specific model:
```bash
python train.py --model rf
python train.py --model svm
```

## Prediction

```bash
python predict.py path/to/image.jpg
```

Prints a single float, e.g.:
```
0.872341
```

## Project Structure

| File | Responsibility |
|---|---|
| `config.py` | All tunable constants (no logic) |
| `utils.py` | Generic image I/O and helper functions |
| `features.py` | All handcrafted feature extraction |
| `classifier.py` | Model factories, training, persistence |
| `train.py` | Dataset loading + training orchestration |
| `evaluate.py` | Metric computation and reporting |
| `predict.py` | CLI entry point for single-image prediction |

## Results

_Fill in after running `train.py` on your collected dataset:_
- Test accuracy:
- ROC-AUC:
- CV accuracy (mean ± std):
- Average prediction latency:
