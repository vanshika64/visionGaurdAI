"""
Handcrafted feature extraction for screen-vs-real classification.

Each function takes a preprocessed (resized) BGR image and returns a dict
of {feature_name: float}. extract_all_features() concatenates everything
into a single fixed-order vector, with FEATURE_NAMES kept in lockstep so
predict.py / evaluate.py can always interpret the vector correctly even if
this file changes later.
"""

from collections import OrderedDict

import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern

import config
from utils import to_gray, safe_divide


# ---------------------------------------------------------------------------
# 1. FFT radial energy-band features
# ---------------------------------------------------------------------------
def fft_features(gray: np.ndarray) -> dict:
    """
    Compute the 2D FFT magnitude spectrum, radially bin it into bands
    (config.FFT_BANDS), and return each band's energy fraction of total
    energy, plus a "deviation from natural 1/f falloff" score.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    power = magnitude ** 2

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_radius = np.sqrt(cx ** 2 + cy ** 2)
    norm_radius = radius / max_radius  # 0 at center, 1 at corner

    total_energy = power.sum()
    feats = {}
    band_energies = []
    for i, (lo, hi) in enumerate(config.FFT_BANDS):
        mask = (norm_radius >= lo) & (norm_radius < hi)
        band_energy = power[mask].sum()
        frac = safe_divide(band_energy, total_energy)
        feats[f"fft_band_{i}_energy_frac"] = frac
        band_energies.append(frac)

    # Radially averaged 1D profile, used to measure deviation from the
    # smooth monotonic decay expected in natural images.
    n_bins = 40
    bin_idx = np.clip((norm_radius * n_bins).astype(int), 0, n_bins - 1)
    radial_profile = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    np.add.at(radial_profile, bin_idx.ravel(), power.ravel())
    np.add.at(counts, bin_idx.ravel(), 1.0)
    radial_profile = radial_profile / np.maximum(counts, 1)
    radial_profile_log = np.log1p(radial_profile)

    # Fit a line (in log space, skip the DC-heavy first couple of bins) and
    # measure residual variance -- natural images decay near-linearly in
    # log-log space; screen photos show spikes that break this linearity.
    valid = radial_profile_log[2:]
    if len(valid) > 2 and valid.std() > 0:
        xs = np.arange(len(valid))
        coeffs = np.polyfit(xs, valid, 1)
        fitted = np.polyval(coeffs, xs)
        residual = valid - fitted
        feats["fft_falloff_residual_std"] = float(residual.std())
        feats["fft_falloff_residual_max"] = float(np.max(np.abs(residual)))
    else:
        feats["fft_falloff_residual_std"] = 0.0
        feats["fft_falloff_residual_max"] = 0.0

    # High/low band ratio, a simple proxy used alongside the per-band split
    feats["fft_high_low_ratio"] = safe_divide(
        band_energies[-1], band_energies[0] + 1e-8
    )

    return feats


# ---------------------------------------------------------------------------
# 2. Moire pattern detection (paired off-axis peaks in FFT)
# ---------------------------------------------------------------------------
def moire_score(gray: np.ndarray) -> dict:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_radius = np.sqrt(cx ** 2 + cy ** 2)
    norm_radius = radius / max_radius

    band_lo, band_hi = config.MOIRE_BAND_RADIUS
    band_mask = (norm_radius >= band_lo) & (norm_radius < band_hi) & (
        norm_radius >= config.MOIRE_LOW_FREQ_RADIUS
    )

    band_values = magnitude[band_mask]
    if band_values.size == 0:
        return {"moire_peak_count": 0.0, "moire_peak_energy_frac": 0.0}

    thresh = np.percentile(band_values, config.MOIRE_PEAK_PERCENTILE)
    peak_mask = (magnitude > thresh) & band_mask

    # Count connected peak blobs (moire shows up as a handful of sharp,
    # localized, often symmetric spikes -- not diffuse noise).
    peak_mask_uint8 = (peak_mask.astype(np.uint8)) * 255
    num_labels, _ = cv2.connectedComponents(peak_mask_uint8, connectivity=8)
    peak_count = max(num_labels - 1, 0)  # subtract background label

    peak_energy = magnitude[peak_mask].sum()
    band_energy = magnitude[band_mask].sum()
    peak_energy_frac = safe_divide(peak_energy, band_energy)

    return {
        "moire_peak_count": float(peak_count),
        "moire_peak_energy_frac": float(peak_energy_frac),
    }


# ---------------------------------------------------------------------------
# 3. Wavelet sub-band energy features
# ---------------------------------------------------------------------------
def wavelet_features(gray: np.ndarray) -> dict:
    coeffs = pywt.wavedec2(
        gray.astype(np.float32), config.WAVELET_NAME, level=config.WAVELET_LEVELS
    )
    # coeffs[0] = final approximation, coeffs[1:] = (cH, cV, cD) per level,
    # ordered from coarsest to finest level.
    feats = {}
    energies = []
    for level_idx, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
        eH = float(np.sum(cH ** 2))
        eV = float(np.sum(cV ** 2))
        eD = float(np.sum(cD ** 2))
        feats[f"wavelet_l{level_idx}_diag_energy"] = eD
        energies.append((level_idx, eH, eV, eD))

    total_energy = sum(eH + eV + eD for _, eH, eV, eD in energies) + 1e-8
    for level_idx, eH, eV, eD in energies:
        feats[f"wavelet_l{level_idx}_diag_frac"] = safe_divide(eD, total_energy)

    # Ratio of diagonal energy between levels -- a proxy for the
    # scale-inconsistent energy moire injects.
    if len(energies) >= 2:
        d1 = energies[0][3]
        d2 = energies[1][3]
        feats["wavelet_diag_level_ratio"] = safe_divide(d2, d1 + 1e-8)
    else:
        feats["wavelet_diag_level_ratio"] = 0.0

    return feats


# ---------------------------------------------------------------------------
# 4. Edge density / Hough line straightness (bezel detection)
# ---------------------------------------------------------------------------
def edge_features(gray: np.ndarray) -> dict:
    edges = cv2.Canny(gray, config.CANNY_LOW, config.CANNY_HIGH)
    edge_density = float(np.mean(edges > 0))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LENGTH,
        maxLineGap=config.HOUGH_MAX_LINE_GAP,
    )

    axis_aligned_count = 0
    total_lines = 0
    total_length = 0.0
    axis_aligned_length = 0.0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angle = abs(angle) % 180
            dist_to_axis = min(angle, abs(angle - 90), abs(angle - 180))
            total_lines += 1
            total_length += length
            if dist_to_axis <= config.AXIS_ALIGN_TOLERANCE_DEG:
                axis_aligned_count += 1
                axis_aligned_length += length

    return {
        "edge_density": edge_density,
        "hough_line_count": float(total_lines),
        "hough_axis_aligned_frac": safe_divide(axis_aligned_count, total_lines),
        "hough_axis_aligned_length_frac": safe_divide(
            axis_aligned_length, total_length
        ),
    }


# ---------------------------------------------------------------------------
# 5. Local Binary Pattern texture summary
# ---------------------------------------------------------------------------
def lbp_features(gray: np.ndarray) -> dict:
    lbp = local_binary_pattern(
        gray, config.LBP_POINTS, config.LBP_RADIUS, method=config.LBP_METHOD
    )
    n_bins = config.LBP_POINTS + 2  # uniform method bin count
    hist, _ = np.histogram(
        lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
    )
    hist = hist + 1e-8
    entropy = float(-np.sum(hist * np.log2(hist)))
    top_bin_frac = float(np.max(hist))
    uniform_frac = float(np.sum(hist[:-1]))  # all but the "non-uniform" bin

    return {
        "lbp_entropy": entropy,
        "lbp_top_bin_frac": top_bin_frac,
        "lbp_uniform_frac": uniform_frac,
    }


# ---------------------------------------------------------------------------
# 6. Gradient orientation peakiness
# ---------------------------------------------------------------------------
def gradient_orientation_features(gray: np.ndarray) -> dict:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = (np.degrees(np.arctan2(gy, gx)) % 180)

    # Weight orientation histogram by magnitude so flat regions don't dilute
    # the signal.
    strong_mask = magnitude > (magnitude.mean() + magnitude.std())
    if strong_mask.sum() == 0:
        return {"gradient_cardinal_peak_frac": 0.0}

    angles = orientation[strong_mask]
    near_cardinal = (
        (angles <= 5) | (angles >= 175) | (np.abs(angles - 90) <= 5)
    )
    frac = float(np.mean(near_cardinal))
    return {"gradient_cardinal_peak_frac": frac}


# ---------------------------------------------------------------------------
# 7. Laplacian variance (sharpness proxy)
# ---------------------------------------------------------------------------
def laplacian_variance(gray: np.ndarray) -> dict:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {"laplacian_variance": float(lap.var())}


# ---------------------------------------------------------------------------
# 8. Color / saturation statistics
# ---------------------------------------------------------------------------
def color_features(bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    # Flat-region uniformity: segment low-gradient regions and measure how
    # unnaturally uniform their color is (screens often render "too clean"
    # flat colors compared to photographed real surfaces).
    gray = to_gray(bgr)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    flat_mask = grad_mag < np.percentile(grad_mag, 20)

    if flat_mask.sum() > 10:
        flat_std = float(bgr[flat_mask].astype(np.float32).std())
    else:
        flat_std = 0.0

    return {
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "high_saturation_frac": float(np.mean(saturation > 0.6)),
        "flat_region_color_std": flat_std,
    }


# ---------------------------------------------------------------------------
# 9. Brightness distribution
# ---------------------------------------------------------------------------
def brightness_features(gray: np.ndarray) -> dict:
    hist, _ = np.histogram(gray.ravel(), bins=32, range=(0, 256), density=True)
    hist = hist + 1e-8
    entropy = float(-np.sum(hist * np.log2(hist)))
    clipped_highlight_frac = float(np.mean(gray > 250))
    dynamic_range = float(gray.max()) - float(gray.min())

    return {
        "brightness_entropy": entropy,
        "clipped_highlight_frac": clipped_highlight_frac,
        "dynamic_range": dynamic_range,
    }


# ---------------------------------------------------------------------------
# 10. Reflection / specular highlight detection
# ---------------------------------------------------------------------------
def reflection_score(bgr: np.ndarray, gray: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    bright_mask = (gray > config.REFLECTION_BRIGHTNESS_THRESH) & (saturation < 60)
    bright_mask_uint8 = bright_mask.astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright_mask_uint8, connectivity=8
    )

    total_pixels = gray.shape[0] * gray.shape[1]
    valid_blob_count = 0
    valid_blob_area = 0
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        area_frac = area / total_pixels
        if area >= config.REFLECTION_MIN_AREA and area_frac <= config.REFLECTION_MAX_AREA_FRACTION:
            valid_blob_count += 1
            valid_blob_area += area

    return {
        "reflection_blob_count": float(valid_blob_count),
        "reflection_area_frac": safe_divide(valid_blob_area, total_pixels),
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def extract_all_features(bgr_image: np.ndarray) -> "OrderedDict[str, float]":
    """
    Run every feature family on a preprocessed (already resized) BGR image
    and return a single ordered dict of feature_name -> value. The ordering
    is deterministic (insertion order of this function), and FEATURE_NAMES
    below must always match it.
    """
    gray = to_gray(bgr_image)

    all_feats = OrderedDict()
    all_feats.update(fft_features(gray))
    all_feats.update(moire_score(gray))
    all_feats.update(wavelet_features(gray))
    all_feats.update(edge_features(gray))
    all_feats.update(lbp_features(gray))
    all_feats.update(gradient_orientation_features(gray))
    all_feats.update(laplacian_variance(gray))
    all_feats.update(color_features(bgr_image))
    all_feats.update(brightness_features(gray))
    all_feats.update(reflection_score(bgr_image, gray))

    return all_feats


def features_to_vector(feats: "OrderedDict[str, float]") -> np.ndarray:
    return np.array(list(feats.values()), dtype=np.float64)


def get_feature_names() -> list:
    """
    Returns the canonical, ordered feature name list by running extraction
    on a tiny dummy image. Used at training time to persist the schema.
    """
    dummy = np.random.randint(0, 255, (config.RESIZE_DIM, config.RESIZE_DIM, 3), dtype=np.uint8)
    feats = extract_all_features(dummy)
    return list(feats.keys())
