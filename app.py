
import streamlit as st
import torch
import torch.nn.functional as F
import streamlit.components.v1 as components
import json

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    CLIPProcessor,
    CLIPModel
)
from PIL import Image

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
        <h2>🐟 15</h2><p>Fish Species</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h2>🎯 82%</h2><p>Test Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h2>🧠 ViT + CLIP</h2><p>AI Image Checking</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- LOAD ORIGINAL FISH MODEL ----------------
MODEL_ID = "BismaLJ/fish_species_classification"

@st.cache_resource
def load_fish_model():
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

# ---------------- LOAD SEPARATE IMAGE CHECKER ----------------
CLIP_ID = "openai/clip-vit-base-patch32"

@st.cache_resource
def load_clip_model():
    processor = CLIPProcessor.from_pretrained(CLIP_ID)
    model = CLIPModel.from_pretrained(CLIP_ID)
    model.eval()
    return processor, model

try:
    with st.spinner("Loading AquaVision AI models..."):
        fish_processor, fish_model = load_fish_model()
        clip_processor, clip_model = load_clip_model()

    st.success("AquaVision AI is ready!")

except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()

# ---------------- CHECK WHETHER IMAGE IS A FISH ----------------
def check_if_fish(image):
    candidate_texts = [
        "a photograph of a fish",
        "a photograph of something that is not a fish"
    ]

    inputs = clip_processor(
        text=candidate_texts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.inference_mode():
        outputs = clip_model(**inputs)
        scores = outputs.logits_per_image.softmax(dim=1)[0]

    fish_score = scores[0].item()
    non_fish_score = scores[1].item()

    return fish_score, non_fish_score

# ---------------- SPEECH FUNCTION ----------------
def speak_result(message):
    safe_message = json.dumps(message)

    html = f"""
    <div style="text-align:center;">
        <button onclick='
            window.speechSynthesis.cancel();
            var speech = new SpeechSynthesisUtterance({safe_message});
            speech.lang = "en-US";
            speech.rate = 0.9;
            window.speechSynthesis.speak(speech);
        '
        style="
            background: linear-gradient(90deg,#00c9a7,#00a6a6);
            color:white;
            border:none;
            padding:12px 25px;
            border-radius:10px;
            font-size:17px;
            font-weight:bold;
            cursor:pointer;">
            🔊 Speak Result
        </button>
    </div>
    """

    components.html(html, height=65)

# ---------------- IMAGE INPUT ----------------
st.markdown("## 📸 Upload or Capture an Image")
st.write("Choose an image from your device or use your camera.")

input_method = st.radio(
    "Select image source:",
    ["Upload Image", "Take a Photo"],
    horizontal=True
)

if input_method == "Upload Image":
    image_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Capture an image")

# ---------------- PREDICTION ----------------
if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")

        st.markdown("### 🖼️ Your Selected Image")
        st.image(
            image,
            caption="Image ready for analysis",
            width=400
        )

        if st.button("✨ Predict Fish Species"):

            with st.spinner("Checking image and analyzing..."):

                # Step 1: Check if image is a fish
                fish_score, non_fish_score = check_if_fish(image)

                st.markdown("---")

                # Step 2: Reject images that do not look like fish
                # This threshold is a starting point and needs testing.
                if fish_score < 0.60 or fish_score <= non_fish_score:

                    st.error("❌ This is not recognized as a fish image.")

                    st.write(
                        f"Fish image score: {fish_score * 100:.2f}%"
                    )

                    st.write(
                        f"Non-fish image score: {non_fish_score * 100:.2f}%"
                    )

                    st.info(
                        "Please upload a clear image of a fish."
                    )

                    speak_result("This is not recognized as a fish image. Please upload a fish image.")

                else:
                    # Step 3: Use the original fish classification model
                    inputs = fish_processor(
                        images=image,
                        return_tensors="pt"
                    )

                    with torch.inference_mode():
                        outputs = fish_model(**inputs)
                        probabilities = F.softmax(
                            outputs.logits,
                            dim=-1
                        )[0]

                    top_values, top_indices = torch.topk(
                        probabilities,
                        5
                    )

                    predicted_index = top_indices[0].item()
                    predicted_label = fish_model.config.id2label[
                        predicted_index
                    ]
                    confidence = top_values[0].item() * 100

                    st.markdown("## 🏆 Classification Result")

                    st.markdown(f"""
                    <div class="result-card">
                        <h2>🐠 {predicted_label}</h2>
                        <h3>Confidence: {confidence:.2f}%</h3>
                        <p>Our AI model's most likely prediction</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Step 4: Speak fish name
                    speak_result(
                        f"The detected fish species is {predicted_label}. "
                        f"Confidence is {confidence:.1f} percent."
                    )

                    # Step 5: Show top 5 predictions
                    st.markdown("### 📊 Top 5 Predictions")

                    for probability, index in zip(
                        top_values,
                        top_indices
                    ):
                        label = fish_model.config.id2label[index.item()]
                        percentage = probability.item() * 100

                        st.write(
                            f"**{label} — {percentage:.2f}%**"
                        )
                        st.progress(float(probability.item()))

                    st.caption(
                        "The fish classifier predicts among its 15 trained "
                        "species. Predictions may be incorrect."
                    )

    except Exception as e:
        st.error(f"Error processing image: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:20px; color:#caf0f8;">
    <h4>🌊 AquaVision AI</h4>
    <p>Powered by Vision Transformer and CLIP</p>
    <p>Built with Python, PyTorch and Streamlit</p>
    <p>Fish Species Classification Project</p>
</div>
""", unsafe_allow_html=True)


