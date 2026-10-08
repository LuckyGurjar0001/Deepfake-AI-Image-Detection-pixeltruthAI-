# Pixel Truth AI - Deepfake / AI-Image Detector

Streamlit app (dark UI) that estimates whether an uploaded image is a **real camera photo** or **AI-generated**.

## Folder layout
```
PixelTruth_AI/
├── deepstream.py            # Streamlit UI  (run this)
├── pixeldetect.py           # preprocessing, features, model loading, scoring, plots
├── train.py                 # trains the classifier from your labelled images
├── model.json               # trained model (created by train.py)
├── requirements.txt         # app libraries (used by Streamlit Cloud)
├── requirements-train.txt   # adds scikit-learn, only needed for train.py
├── .streamlit/config.toml   # dark theme
├── dataset/train/real/      # camera photos used for training
├── dataset/train/fake/      # AI-generated images used for training
└── sample_images/           # real_camera_photo.jpg and ai_generated_portrait.png
```

## Run locally (Windows / VS Code)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run deepstream.py
```
Open http://localhost:8501 and upload an image (try the two files in `sample_images/`).

## How it works
1. Every image is normalised the same way (RGB, resize, centre-crop 512x512, JPEG re-encode) so file format does not decide the result.
2. Features are measured: noise in flat areas, frequency-band energy, chroma noise, detail, edges, colour histograms.
3. A scikit-learn logistic-regression classifier (`model.json`, plain NumPy at run time) turns them into an AI probability.
4. The UI shows the verdict, score, measured values, and six evidence plots.

If `model.json` is missing, the app falls back to a simple heuristic and says so on screen.

## Training your own model
1. Put camera photos in `dataset/train/real/` and AI images in `dataset/train/fake/`.
2. `pip install -r requirements-train.txt` (once), then run `python train.py`.
3. Restart the app.

`train.py` prints cross-validated accuracy and a confusion matrix once you have at least 5 images per class.

## Important limitations
- The model that ships in this repo was trained on **only 2 images** (one real, one AI). It is a **demo**: it gives the expected result on those two samples because it has seen them, and it can be wrong on other images. The app shows a warning banner in this case.
- For results you can trust, train on 100+ images per class and read the cross-validated accuracy.
- No detector is perfect. Treat results as an indication, not proof.
