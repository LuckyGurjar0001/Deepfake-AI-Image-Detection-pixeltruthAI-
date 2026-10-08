"""
train.py  -  Train the Pixel Truth AI classifier from your own labelled images.

Put images here, then run:   python train.py
    dataset/train/real/   photos taken by a camera
    dataset/train/fake/   AI-generated images

The result is saved to model.json, which the Streamlit app loads automatically.
More (and more varied) images = a more trustworthy model. Aim for 100+ per class.
"""
from pathlib import Path

import json

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pixeldetect import FEATURE_NAMES, MODEL_PATH, RELIABLE_MIN_PER_CLASS, extract_features_v2, normalise_image

ROOT = Path(__file__).parent / "dataset" / "train"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_folder(folder: Path):
    X, names = [], []
    for p in sorted(folder.glob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        try:
            img = Image.open(p)
            img.load()
            X.append(extract_features_v2(normalise_image(img)))
            names.append(p.name)
        except Exception as exc:
            print(f"  skipped {p.name}: {exc}")
    return X, names


def main():
    real_X, real_names = load_folder(ROOT / "real")
    fake_X, fake_names = load_folder(ROOT / "fake")
    n_r, n_f = len(real_X), len(fake_X)
    print(f"Loaded {n_r} real and {n_f} AI-generated images")
    if n_r == 0 or n_f == 0:
        raise SystemExit("Need at least one image in BOTH dataset/train/real and dataset/train/fake.")

    X = np.array(real_X + fake_X)
    y = np.array([0] * n_r + [1] * n_f)          # 0 = real camera photo, 1 = AI-generated
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))

    cv_acc, cv_cm = None, None
    smallest = min(n_r, n_f)
    if smallest >= 5:
        folds = min(5, smallest)
        pred = cross_val_predict(pipe, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=42))
        cv_acc = float(accuracy_score(y, pred))
        cv_cm = confusion_matrix(y, pred).tolist()
        print(f"\n{folds}-fold cross-validated accuracy: {cv_acc:.1%}")
        print("Confusion matrix (rows = truth real/AI, cols = predicted real/AI):")
        print(np.array(cv_cm))
    else:
        print("\nToo few images for cross-validation (need at least 5 per class), so NO honest accuracy can be reported.")

    pipe.fit(X, y)
    print(f"Accuracy on the training images themselves: {pipe.score(X, y):.1%}  (not a real measure of quality)")

    scaler = pipe.named_steps["standardscaler"]
    clf = pipe.named_steps["logisticregression"]
    assert list(clf.classes_) == [0, 1]
    bundle = {
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
        "n_real": n_r, "n_fake": n_f, "cv_accuracy": cv_acc, "cv_confusion": cv_cm,
        "files_real": real_names, "files_fake": fake_names,
    }
    MODEL_PATH.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nSaved {MODEL_PATH.name}")
    if n_r < RELIABLE_MIN_PER_CLASS or n_f < RELIABLE_MIN_PER_CLASS:
        print(f"WARNING: fewer than {RELIABLE_MIN_PER_CLASS} images per class. The app will label this a demo model.")


if __name__ == "__main__":
    main()
