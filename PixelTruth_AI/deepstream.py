"""
deepstream.py  -  Pixel Truth AI  (Streamlit front-end, dark UI)

Run with:   streamlit run deepstream.py
"""
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from pixeldetect import analyze_image, build_analysis_figure

st.set_page_config(
    page_title="Pixel Truth AI",
    page_icon="🔍",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Dark UI styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
    .stApp { background: radial-gradient(circle at 20% 0%, #151c30 0%, #0B0F19 55%); }
    header[data-testid="stHeader"] { background: transparent; }

    .main-header {
        font-size: 3.4rem; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #8B5CF6, #38BDF8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0.2rem 0 0.2rem 0;
    }
    .sub-header {
        font-size: 1.15rem; color: #9CA3AF; text-align: center; margin-bottom: 1.6rem;
    }
    .info-box {
        background: #131A2A; padding: 22px 26px; border-radius: 12px;
        border-left: 5px solid #8B5CF6; margin-bottom: 1.2rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }
    .info-box h2, .info-box h3 { color: #F3F4F6; margin-top: 0; }
    .info-box p, .info-box li { color: #CBD5E1; }

    .upload-box {
        background: #131A2A; border: 2px dashed #8B5CF6; border-radius: 12px;
        padding: 26px; text-align: center; margin-bottom: 0.6rem;
    }
    .upload-box h2 { color: #F3F4F6; margin: 0 0 6px 0; }
    .upload-box p  { color: #9CA3AF; margin: 0; }

    .result-box {
        padding: 26px; border-radius: 14px; text-align: center; margin: 1rem 0 1.4rem 0;
        box-shadow: 0 6px 24px rgba(0,0,0,0.45);
    }
    .result-box h2 { margin: 0 0 8px 0; color: #F9FAFB; }
    .result-box p  { margin: 4px 0; color: #E5E7EB; }
    .result-fake      { background: rgba(239,68,68,0.14);  border: 1.5px solid #EF4444; }
    .result-real      { background: rgba(34,197,94,0.14);  border: 1.5px solid #22C55E; }
    .result-uncertain { background: rgba(245,158,11,0.14); border: 1.5px solid #F59E0B; }

    .score-track { background: #1F2937; border-radius: 999px; height: 14px; overflow: hidden; margin: 10px 0 2px 0; }
    .score-fill  { height: 100%; border-radius: 999px; }

    .metric-card {
        background: #131A2A; border: 1px solid #1F2937; border-radius: 12px;
        padding: 14px 16px; text-align: center;
    }
    .metric-card .label { color: #9CA3AF; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .05em; }
    .metric-card .value { color: #F3F4F6; font-size: 1.5rem; font-weight: 700; }

    .trust-box { padding: 14px 18px; border-radius: 12px; margin: 0 0 1rem 0; font-size: 0.95rem; }
    .trust-box b { color: #F9FAFB; }
    .trust-warn { background: rgba(245,158,11,0.12); border: 1px solid #F59E0B; color: #FDE68A; }
    .trust-ok   { background: rgba(34,197,94,0.10);  border: 1px solid #22C55E; color: #BBF7D0; }
    .trust-info { background: rgba(56,189,248,0.10); border: 1px solid #38BDF8; color: #BAE6FD; }

    .section-title { color: #F3F4F6; font-size: 1.7rem; font-weight: 700; margin: 1.2rem 0 0.6rem 0; }
    hr { border-color: #1F2937; }

    [data-testid="stFileUploader"] section {
        background: #131A2A; border: 1px solid #1F2937; border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="main-header">Pixel Truth AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Deepfake &amp; AI-generated image detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="info-box">
    <h2>Welcome to Pixel Truth AI</h2>
    <p>This application helps detect AI-generated or deepfake images using advanced image
    analysis techniques. Upload an image to analyze whether it's likely real or AI-generated.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="upload-box">
    <h2>Upload an Image to Analyze if it is Real or Virtual</h2>
    <p>Supported formats: JPG, JPEG, PNG, WEBP</p>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

# ----------------------------------------------------------------------------
# Analysis + results
# ----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.load()
    except Exception as exc:  # corrupted / unsupported file
        st.error(f"Could not read this image: {exc}")
        st.stop()

    with st.spinner("Analyzing image..."):
        result = analyze_image(image)

    score = result["score"]
    kind = result["kind"]
    fill_color = {"fake": "#EF4444", "real": "#22C55E", "uncertain": "#F59E0B"}[kind]
    descriptions = {
        "fake": "This image shows characteristics consistent with AI-generated content.",
        "real": "This image shows characteristics consistent with a real photograph.",
        "uncertain": "Signals are mixed. The image could be either real or AI-generated.",
    }

    # --- how much should the user trust this result? -------------------------------------
    info = result["model_info"]
    if result["mode"] == "model" and info and info["reliable"]:
        banner = ("trust-ok", f"<b>Trained classifier</b> ({info['n_real']} real + {info['n_fake']} AI training images). "
                              f"Cross-validated accuracy on those images: <b>{info['cv_accuracy']:.0%}</b>. "
                              "Real-world accuracy can still be lower on image types it has not seen.")
    elif result["mode"] == "model" and info:
        banner = ("trust-warn", f"<b>Demo model - not reliable.</b> It was trained on only {info['n_real']} real and "
                                f"{info['n_fake']} AI image(s), so it has effectively memorised those samples. "
                                "Results on other images can be wrong. Add more images to <code>dataset/train</code> "
                                "and run <code>python train.py</code>.")
    else:
        banner = ("trust-warn", "<b>Basic heuristic mode - not reliable.</b> No trained model was found. This simple "
                                "rule (smooth, noise-free = AI) often fails on sharp or compressed photos. "
                                "Run <code>python train.py</code> to train a model.")
    st.markdown(f'<div class="trust-box {banner[0]}">{banner[1]}</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown('<div class="section-title">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.caption(f"{uploaded_file.name}  •  {image.size[0]}×{image.size[1]} px")

    with right:
        st.markdown(
            f"""
<div class="result-box result-{kind}">
    <h2>Result: {result['label']}</h2>
    <p>AI Detection Score: <b>{score:.2f}%</b></p>
    <div class="score-track"><div class="score-fill" style="width:{score:.1f}%; background:{fill_color};"></div></div>
    <p>{descriptions[kind]}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Signal Breakdown</div>', unsafe_allow_html=True)
        cols = st.columns(len(result["parts"]))
        for col, (name, val) in zip(cols, result["parts"].items()):
            col.markdown(
                f'<div class="metric-card"><div class="label">{name}</div>'
                f'<div class="value">{val}</div></div>',
                unsafe_allow_html=True,
            )

        f = result["features"]
        st.markdown('<div class="section-title">Measured Features</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Edge density", f"{f['edge_density']:.3f}")
        m2.metric("Noise level", f"{f['noise_level']:.2f}")
        m3.metric("Texture variance", f"{f['texture_var']:.1f}")
        m4.metric("Hist. smoothness", f"{f['hist_smoothness']:.3f}")

    st.markdown('<div class="section-title">Feature Analysis</div>', unsafe_allow_html=True)
    fig = build_analysis_figure(result["display_array"])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if result["mode"] == "model":
        analysed = """
        <li><b>Noise in flat areas:</b> camera sensors leave fine grain even on a plain wall; generators often do not</li>
        <li><b>Frequency content:</b> how much energy sits in fine detail compared with coarse shapes</li>
        <li><b>Colour noise:</b> the grain pattern in the colour channels</li>
        <li><b>Edges, texture and colour histograms:</b> sharpness, detail and smoothness</li>
        <li>Every image is resized, cropped and re-compressed the same way first, so file format does not decide the result</li>"""
        intro = "A classifier trained on your labelled images compared these measurements:"
    else:
        analysed = """
        <li><b>Color distribution:</b> AI images often have smoother color histograms</li>
        <li><b>Edge consistency:</b> AI images may have fewer or unnatural edges</li>
        <li><b>Noise patterns:</b> AI images often have less noise than real photos</li>
        <li><b>Texture artifacts:</b> AI images may have repetitive or uniform textures</li>"""
        intro = "The application examined various aspects of your image:"
    st.markdown(
        f"""
<div class="info-box">
    <h3>What We Analyzed</h3>
    <p>{intro}</p>
    <ul>{analysed}
    </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(
        "⚠️ AI-image detection is hard and no tool is perfect. Treat this result as an indication, not proof."
    )
