"""
pixeldetect.py  -  Image analysis engine for Pixel Truth AI

Contains the same pipeline shown in the project report:
  1. preprocess_image()  -> resize + drop alpha channel
  2. extract_features()  -> colour histograms, Canny edges, noise, texture
  3. compute_ai_score()  -> combine features into an AI-likelihood score
  4. build_analysis_figure() -> dark-themed feature visualisation
"""
from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).with_name("model.json")   # plain JSON: needs only NumPy, no scikit-learn version issues
NORM_SIDE = 512              # every image is normalised to 512x512 before ML features are measured
RELIABLE_MIN_PER_CLASS = 20  # below this, the app warns that the trained model is only a demo

DISPLAY_SIZE = (50, 50)      # small size used for the visual panels (as in report screenshots)
ANALYSIS_SIZE = (224, 224)   # size used for numeric feature extraction


# ----------------------------------------------------------------------------
# 1. Pre-processing
# ----------------------------------------------------------------------------
def preprocess_image(img: Image.Image, size: tuple[int, int] = DISPLAY_SIZE) -> np.ndarray:
    """Resize to a standard size and strip any transparency channel."""
    img = img.convert("RGB")                 # handles RGBA / P / L modes safely
    img = img.resize(size)
    img_array = np.array(img)
    if img_array.shape[-1] == 3:             # keep only R, G, B
        img_array = img_array[:, :, :3]
    return img_array


# ----------------------------------------------------------------------------
# 2. Feature extraction
# ----------------------------------------------------------------------------
def _hist_smoothness(hist: np.ndarray) -> float:
    """Mean absolute change between neighbouring histogram bins (lower = smoother)."""
    h = hist.flatten().astype(np.float64)
    h = h / (h.sum() + 1e-9)
    return float(np.mean(np.abs(np.diff(h))) * 256)


def extract_features(img_array: np.ndarray) -> dict:
    """Extract colour, edge, noise and texture features from an RGB array."""
    features: dict = {}
    h, w = img_array.shape[:2]

    # Colour spaces
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Colour histograms
    features["r_hist"] = cv2.calcHist([img_array], [0], None, [256], [0, 256])
    features["g_hist"] = cv2.calcHist([img_array], [1], None, [256], [0, 256])
    features["b_hist"] = cv2.calcHist([img_array], [2], None, [256], [0, 256])
    features["hist_smoothness"] = float(np.mean([
        _hist_smoothness(features["r_hist"]),
        _hist_smoothness(features["g_hist"]),
        _hist_smoothness(features["b_hist"]),
    ]))

    # Edge features (Canny 100/200)
    edges = cv2.Canny(gray, 100, 200)
    features["edge_density"] = float(np.sum(edges > 0) / (h * w))

    # Noise: residual after Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    features["noise_level"] = float(np.std(residual))

    # Texture: Laplacian variance (sharpness / micro-detail)
    features["texture_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Saturation statistics
    features["saturation_mean"] = float(np.mean(hsv[:, :, 1]))
    features["saturation_std"] = float(np.std(hsv[:, :, 1]))

    return features


# ----------------------------------------------------------------------------
# 3. Scoring
# ----------------------------------------------------------------------------
def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def compute_ai_score(features: dict) -> tuple[float, dict]:
    """
    Combine features into an AI-likelihood score (0-100).

    Heuristics (match the "What We Analyzed" section of the app):
      * smooth colour histograms   -> more AI-like
      * low sensor noise           -> more AI-like
      * smooth / uniform texture   -> more AI-like
      * unnatural edge density     -> more AI-like
      * unusually even saturation  -> more AI-like
    """
    smooth_score = _clip01(1.0 - features["hist_smoothness"] / 0.9)
    noise_score = _clip01(1.0 - features["noise_level"] / 6.0)
    texture_score = _clip01(1.0 - features["texture_var"] / 400.0)
    ed = features["edge_density"]
    edge_score = _clip01(abs(ed - 0.12) / 0.12) if ed < 0.12 else _clip01((ed - 0.12) / 0.30)
    sat_score = _clip01(1.0 - features["saturation_std"] / 70.0)

    parts = {
        "Colour smoothness": smooth_score,
        "Low noise": noise_score,
        "Texture uniformity": texture_score,
        "Edge irregularity": edge_score,
        "Even saturation": sat_score,
    }
    weights = {
        "Colour smoothness": 0.25,
        "Low noise": 0.30,
        "Texture uniformity": 0.20,
        "Edge irregularity": 0.10,
        "Even saturation": 0.15,
    }
    score = sum(parts[k] * weights[k] for k in parts) * 100.0
    return float(score), {k: round(v * 100, 1) for k, v in parts.items()}


def classify(score: float) -> tuple[str, str]:
    """Return (label, css_class) for a given AI score."""
    if score >= 60:
        return "Likely AI-Generated", "fake"
    if score >= 40:
        return "Uncertain / Mixed Signals", "uncertain"
    return "Likely Real", "real"


# ----------------------------------------------------------------------------
# 3b. Trained-model path (learns from labelled images in dataset/train)
# ----------------------------------------------------------------------------
FEATURE_NAMES = [
    "flat_noise_median", "flat_noise_p10", "residual_std", "fft_mid", "fft_high", "fft_vhigh",
    "log_laplacian_var", "chroma_noise", "mean_gradient", "edge_density", "hist_smoothness",
    "saturation_mean", "saturation_std", "residual_kurtosis",
]


def normalise_image(img: Image.Image) -> np.ndarray:
    """
    Make every image comparable before measuring it:
      * convert to RGB, resize shorter side to 512, centre-crop to 512x512
      * re-encode as JPEG (quality 90) so PNG-vs-JPEG / compression history
        does not become the thing the classifier learns.
    """
    img = img.convert("RGB")
    w, h = img.size
    s = NORM_SIDE / min(w, h)
    img = img.resize((max(NORM_SIDE, round(w * s)), max(NORM_SIDE, round(h * s))), Image.LANCZOS)
    w, h = img.size
    left, top = (w - NORM_SIDE) // 2, (h - NORM_SIDE) // 2
    img = img.crop((left, top, left + NORM_SIDE, top + NORM_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def extract_features_v2(rgb: np.ndarray) -> np.ndarray:
    """Noise, frequency, detail, edge and colour statistics of a normalised 512x512 RGB image."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    res = g - cv2.GaussianBlur(g, (0, 0), 1.5)

    # noise measured only in flat patches (lowest 30% gradient), where texture does not hide it
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    P = 32
    patch_noise, patch_grad = [], []
    for y in range(0, g.shape[0] - P + 1, P):
        for x in range(0, g.shape[1] - P + 1, P):
            patch_noise.append(res[y:y + P, x:x + P].std())
            patch_grad.append(gm[y:y + P, x:x + P].mean())
    patch_noise, patch_grad = np.array(patch_noise), np.array(patch_grad)
    flat = patch_noise[patch_grad <= np.percentile(patch_grad, 30)]

    # frequency-band energy (share of spectrum energy in mid / high / very-high bands)
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    r = np.hypot(yy - f.shape[0] / 2, xx - f.shape[1] / 2) / (min(f.shape) / 2)
    total = f[r < 1.0].sum() + 1e-9
    fft_mid = f[(r >= 0.25) & (r < 0.5)].sum() / total
    fft_high = f[(r >= 0.5) & (r < 0.75)].sum() / total
    fft_vhigh = f[(r >= 0.75) & (r < 1.0)].sum() / total

    lap_var = float(cv2.Laplacian(g, cv2.CV_32F).var())

    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    chroma = np.mean([(ycc[..., c] - cv2.GaussianBlur(ycc[..., c], (0, 0), 1.5)).std() for c in (1, 2)])

    edges = cv2.Canny(g.astype(np.uint8), 100, 200)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist_s = np.mean([_hist_smoothness(cv2.calcHist([rgb], [c], None, [256], [0, 256])) for c in range(3)])
    rs = res.std() + 1e-9
    kurt = float(np.mean(((res - res.mean()) / rs) ** 4))

    return np.array([
        np.median(flat), np.percentile(flat, 10), res.std(), fft_mid, fft_high, fft_vhigh,
        np.log1p(lap_var), chroma, gm.mean(), np.sum(edges > 0) / edges.size, hist_s,
        hsv[..., 1].mean(), hsv[..., 1].std(), kurt,
    ], dtype=np.float64)


@lru_cache(maxsize=2)
def _load_cached(path: str, mtime: float):
    with open(path, "r", encoding="utf-8") as fh:
        bundle = json.load(fh)
    for key in ("mean", "scale", "coef"):
        bundle[key] = np.array(bundle[key], dtype=np.float64)
    if len(bundle["coef"]) != len(FEATURE_NAMES):
        raise ValueError("model.json does not match this version of the feature list - retrain with train.py")
    return bundle


def load_model():
    """Return the trained model bundle, or None (no model file, or it cannot be loaded)."""
    try:
        if not MODEL_PATH.exists():
            return None
        return _load_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception:
        return None


def predict_ai_probability(bundle: dict, vec: np.ndarray) -> float:
    """Logistic regression in plain NumPy: sigmoid(w . standardised(x) + b)."""
    z = (vec - bundle["mean"]) / bundle["scale"]
    logit = float(z @ bundle["coef"] + bundle["intercept"])
    return float(1.0 / (1.0 + np.exp(-logit)))


def model_info(bundle: dict | None) -> dict | None:
    if bundle is None:
        return None
    n_r, n_f = bundle["n_real"], bundle["n_fake"]
    return {
        "n_real": n_r, "n_fake": n_f, "cv_accuracy": bundle.get("cv_accuracy"),
        "reliable": bool(n_r >= RELIABLE_MIN_PER_CLASS and n_f >= RELIABLE_MIN_PER_CLASS
                         and bundle.get("cv_accuracy") is not None),
    }


def analyze_image(img: Image.Image) -> dict:
    """Run the full pipeline and return everything the UI needs."""
    display_arr = preprocess_image(img, DISPLAY_SIZE)
    analysis_arr = preprocess_image(img, ANALYSIS_SIZE)
    features = extract_features(analysis_arr)          # simple stats shown in the UI
    bundle = load_model()

    if bundle is not None:
        vec = extract_features_v2(normalise_image(img))
        prob_ai = predict_ai_probability(bundle, vec)
        score = prob_ai * 100.0
        d = dict(zip(FEATURE_NAMES, vec))
        parts = {
            "Flat-area noise": f"{d['flat_noise_median']:.2f}",
            "High-freq energy": f"{d['fft_high']:.3f}",
            "Chroma noise": f"{d['chroma_noise']:.2f}",
            "Fine detail": f"{d['log_laplacian_var']:.2f}",
            "Edge density": f"{d['edge_density']:.3f}",
        }
        mode, info = "model", model_info(bundle)
    else:
        score, raw_parts = compute_ai_score(features)
        parts = {k: f"{v:.0f}%" for k, v in raw_parts.items()}
        mode, info = "heuristic", None

    label, kind = classify(score)
    return {
        "display_array": display_arr,
        "analysis_array": analysis_arr,
        "features": features,
        "score": score,
        "parts": parts,
        "label": label,
        "kind": kind,
        "mode": mode,
        "model_info": info,
    }


# ----------------------------------------------------------------------------
# 4. Visualisation (dark theme)
# ----------------------------------------------------------------------------
BG = "#131A2A"
FG = "#E5E7EB"


def _style_axis(ax, title: str):
    ax.set_title(title, color=FG, fontsize=11, pad=8)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.tick_params(colors="#9CA3AF", labelsize=8)


def build_analysis_figure(display_array: np.ndarray) -> plt.Figure:
    """2x3 grid: processed, grayscale, edges, RGB hist, HSV hist, noise map."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor(BG)

    # Processed image
    axes[0, 0].imshow(display_array)
    _style_axis(axes[0, 0], "Processed Image")
    axes[0, 0].axis("off")

    # Grayscale
    gray = cv2.cvtColor(display_array, cv2.COLOR_RGB2GRAY)
    axes[0, 1].imshow(gray, cmap="gray")
    _style_axis(axes[0, 1], "Grayscale")
    axes[0, 1].axis("off")

    # Edges
    edges = cv2.Canny(gray, 100, 200)
    axes[0, 2].imshow(edges, cmap="gray")
    _style_axis(axes[0, 2], "Edge Detection")
    axes[0, 2].axis("off")

    # RGB histograms
    for i, color in enumerate(("r", "g", "b")):
        hist = cv2.calcHist([display_array], [i], None, [256], [0, 256])
        axes[1, 0].plot(hist, color=color, linewidth=1.3)
    _style_axis(axes[1, 0], "Color Histograms")
    axes[1, 0].set_xlim([0, 256])

    # HSV histograms
    hsv = cv2.cvtColor(display_array, cv2.COLOR_RGB2HSV)
    for i, color in enumerate(("r", "g", "b")):
        axes[1, 1].plot(cv2.calcHist([hsv], [i], None, [256], [0, 256]), color=color, linewidth=1.3)
    _style_axis(axes[1, 1], "HSV Histograms")
    axes[1, 1].set_xlim([0, 256])

    # Noise residual map
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    axes[1, 2].imshow(residual, cmap="magma")
    _style_axis(axes[1, 2], "Noise Residual Map")
    axes[1, 2].axis("off")

    fig.tight_layout(pad=2.0)
    return fig
