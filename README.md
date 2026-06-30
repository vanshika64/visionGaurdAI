# Real Photo vs. Screen Photo Classifier

A lightweight Computer Vision project that determines whether an input image is:

- A real photo of a real-world object
- A photo of a screen (phone, laptop, monitor, or tablet) displaying an image

The model outputs a single probability:

- **0.0** → Definitely a real photo
- **1.0** → Definitely a photo of a screen

---

# Approach

Instead of using a deep CNN, this project uses handcrafted Computer Vision features designed to detect artifacts that naturally occur when photographing a display.

The extracted features include:

- FFT Frequency Analysis
- Moiré Pattern Detection
- Wavelet Features
- Edge Density
- Hough Line Detection
- Local Binary Pattern (LBP)
- Gradient Orientation Features
- Laplacian Variance
- Color & Saturation Statistics
- Brightness Distribution
- Reflection Detection

These features are combined into a feature vector and classified using a **Random Forest Classifier**.

See `features.py` for the complete feature extraction pipeline.

---

# Setup

```bash
pip install -r requirements.txt
```

---

# Dataset

```
dataset/
│
├── real/
│     image001.jpg
│     ...
│     image050.jpg
│
└── screen/
      image001.jpg
      ...
      image050.jpg
```

The dataset contains:

- 50 real-world photos
- 50 photos of screens

Images should include different:

- Lighting conditions
- Camera angles
- Distances
- Devices
- Backgrounds
- Screen brightness levels
- Reflections

---

# Training

```bash
python train.py
```

The training script:

- Loads the dataset
- Extracts handcrafted features
- Trains a Random Forest classifier
- Evaluates performance
- Prints:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
  - Confusion Matrix
- Saves the trained model to:

```
models/model.pkl
```

---

# Prediction

```bash
python predict.py image.jpg
```

Example output:

```
0.872341
```

---

# Project Structure

| File | Responsibility |
|------|----------------|
| `config.py` | Project configuration and constants |
| `utils.py` | Image loading and helper functions |
| `features.py` | Handcrafted feature extraction |
| `classifier.py` | Random Forest training, prediction and model saving |
| `train.py` | Dataset loading and model training |
| `evaluate.py` | Performance evaluation |
| `predict.py` | Predicts whether an image is a real photo or a screen photo |

---

# Results

Update these after training on your dataset.

- Test Accuracy:0.9500
- Precision:0.9091
- Recall:1.0000
- F1 Score:0.9524
- ROC-AUC:0.9800

---

# Technologies Used

- Python 3.11
- OpenCV
- NumPy
- scikit-image
- scikit-learn
- PyWavelets
- Joblib

---

# Model

This project uses a **Random Forest Classifier** trained on handcrafted Computer Vision features to distinguish real-world photographs from photos of digital screens.