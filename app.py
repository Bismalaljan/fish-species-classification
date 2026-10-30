

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import time

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AquaVision AI",
    page_icon="🐠",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #061b2b, #073b4c, #087e8b);
    color: white;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.hero {
    text-align: center;
    padding: 35px 20px;
    border-radius: 22px;
    background: linear-gradient(120deg, #064663, #087e8b, #00a896);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 46px;
    margin-bottom: 8px;
}

.hero p {
    color: #e0fbfc;
    font-size: 18px;
}

.info-card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 15px;
}

.result-card {
    background: linear-gradient(120deg, #00a896, #028090);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-top: 15px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

.result-card h2 {
    color: white;
    font-size: 30px;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 13px;
    background: linear-gradient(90deg, #00c9a7, #00a6a6);
    color: white;
    font-size: 17px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #02e6bd, #00c2c7);
    color: #062b36;
    border: none;
    transform: scale(1.02);
}

h2, h3, h4, p, label {
    color: white !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border: 1px dashed #64dfdf;
    padding: 15px;
    border-radius: 15px;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.10);
    padding: 15px;
    border-radius: 15px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <h1>🐠 AquaVision AI</h1>
    <p>Discover the underwater world with Artificial Intelligence</p>
    <p>Upload a fish image and let Vision Transformer identify its species.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- PROJECT INFORMATION ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h2>🐟 15</h2>
        <p>Fish Species</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h2>🎯 82%</h2>
        <p>Test Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h2>🧠 ViT</h2>
        <p>AI Model</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
MODEL_ID = "BismaLJ/fish_species_classification"

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

try:
    with st.spinner("Loading AquaVision AI model..."):
        processor, model = load_model()
    st.success("AI model is ready!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ---------------- IMAGE INPUT ----------------
st.markdown("## 📸 Upload or Capture a Fish Image")
st.write("Choose an image from your device or use your camera.")

input_method = st.radio(
    "Select image source:",
    ["Upload Image", "Take a Photo"],
    horizontal=True
)

image_file = None

if input_method == "Upload Image":
    image_file = st.file_uploader(
        "Choose a fish image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Capture a fish image")

# ---------------- PREDICTION ----------------
if image_file is not None:
    image = Image.open(image_file).convert("RGB")

    st.markdown("### 🖼️ Your Selected Image")
    st.image(image, caption="Image ready for classification", width=400)

    if st.button("✨ Predict Fish Species"):
        with st.spinner("Analyzing image with Artificial Intelligence..."):
            inputs = processor(images=image, return_tensors="pt")

            with torch.inference_mode():
                outputs = model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)[0]

            top_values, top_indices = torch.topk(probabilities, 5)

        st.markdown("---")
        st.markdown("## 🏆 Classification Result")

        predicted_index = top_indices[0].item()
        predicted_label = model.config.id2label[predicted_index]
        confidence = top_values[0].item() * 100

        st.markdown(f"""
        <div class="result-card">
            <h2>🐠 {predicted_label}</h2>
            <h3>Confidence: {confidence:.2f}%</h3>
            <p>Our AI model's most likely prediction</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Top 5 Predictions")

        for probability, index in zip(top_values, top_indices):
            label = model.config.id2label[index.item()]
            percentage = probability.item() * 100

            st.write(f"**{label} — {percentage:.2f}%**")
            st.progress(float(probability.item()))

        st.caption(
            "Note: The model selects from the 15 fish species it was trained on. "
            "It may still assign a fish label to an image that is not a fish."
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:20px; color:#caf0f8;">
    <h4>🌊 AquaVision AI</h4>
    <p>Powered by Vision Transformer | Built with Python, PyTorch & Streamlit</p>
    <p>Fish Species Classification Project</p>
</div>
""", unsafe_allow_html=True)

