
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="AquaVision AI",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_ID = "BismaLJ/fish_species_classification"

# ---------------- PROFESSIONAL CUSTOM DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, #123c55 0%, transparent 35%),
        linear-gradient(135deg, #061522, #092b3b 55%, #064b55);
    color: #f1f7fa;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    text-align: center;
    padding: 55px 25px 45px;
    border: 1px solid rgba(125, 211, 225, 0.22);
    border-radius: 25px;
    background: linear-gradient(120deg,
        rgba(10, 65, 86, 0.95),
        rgba(0, 137, 145, 0.75));
    margin-bottom: 35px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
}

.hero h1 {
    color: #ffffff;
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 10px;
}

.hero p {
    color: #d5f3f5;
    font-size: 17px;
    margin: 5px 0;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #e8fbff;
    margin: 25px 0 15px;
}

.panel {
    background: rgba(255,255,255,0.065);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
}

.result-card {
    background: linear-gradient(120deg, #087f83, #06a6a0);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 30px 20px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 12px 30px rgba(0,0,0,0.2);
}

.result-card h2 {
    color: white;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-card p {
    color: #e2ffff;
    margin: 0;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 14px;
    background: linear-gradient(90deg, #00bfa6, #00a6b2);
    color: white;
    font-weight: 700;
    font-size: 16px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #00d5b5, #00c1ca);
    color: #062b36;
    border: none;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 1px dashed #54cbd0;
    padding: 18px;
    border-radius: 15px;
}

[data-testid="stCameraInput"] {
    border-radius: 15px;
}

.stProgress > div > div > div {
    background-color: #00c6ad;
}

h2, h3, h4, p, label {
    color: #f1f7fa;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    padding: 15px;
    border-radius: 14px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

.footer {
    text-align: center;
    color: #a9cbd2;
    padding: 30px 10px 5px;
    border-top: 1px solid rgba(255,255,255,0.12);
    margin-top: 45px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">
    <h1>🐠 AquaVision AI</h1>
    <p>Explore the underwater world through Artificial Intelligence.</p>
    <p>Upload or capture a fish image to discover its predicted species.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD EXISTING MODEL ----------------
@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID)

    model.eval()

    return processor, model


try:
    with st.spinner("Preparing your AI experience..."):
        processor, model = load_model()

    st.success("AquaVision AI is ready.")

except Exception as e:
    st.error(f"Unable to load the AI model: {e}")
    st.stop()


# ---------------- IMAGE INPUT SECTION ----------------
st.markdown(
    '<div class="section-title">📷 Image Analysis</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    input_method = st.radio(
        "Choose your image source",
        ["Upload Image", "Use Camera"],
        horizontal=True
    )

    image_file = None

    if input_method == "Upload Image":
        image_file = st.file_uploader(
            "Select an image",
            type=["jpg", "jpeg", "png"]
        )
    else:
        image_file = st.camera_input("Capture an image")

    st.markdown(
        "<p style='color:#a9cbd2;font-size:13px;'>"
        "Supported formats: JPG, JPEG and PNG"
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


with right:
    if image_file is not None:
        image = Image.open(image_file).convert("RGB")

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Selected image**")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="panel" style="
            min-height: 250px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            text-align:center;">
            <div style="font-size:55px;">🌊</div>
            <h3>Ready to explore?</h3>
            <p style="color:#a9cbd2;">
                Your selected image will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- PREDICTION ----------------
if image_file is not None:

    st.markdown("---")

    if st.button("✨ Analyze Image", use_container_width=True):

        try:
            with st.spinner("Analyzing image with Vision Transformer..."):

                inputs = processor(
                    images=image,
                    return_tensors="pt"
                )

                with torch.inference_mode():
                    outputs = model(**inputs)

                    probabilities = F.softmax(
                        outputs.logits,
                        dim=-1
                    )[0]

                top_values, top_indices = torch.topk(
                    probabilities, 5
                )

            predicted_index = top_indices[0].item()

            predicted_label = model.config.id2label[
                predicted_index
            ]

            confidence = top_values[0].item() * 100

            # ---------------- MAIN RESULT ----------------
            st.markdown("---")

            st.markdown(
                '<div class="section-title">Analysis Result</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:42px;">🐟</div>
                <h2>{predicted_label}</h2>
                <p>Predicted Fish Species</p>
                <h3 style="color:white;">
                    {confidence:.2f}% confidence
                </h3>
            </div>
            """, unsafe_allow_html=True)

            # ---------------- TOP PREDICTIONS ----------------
            st.markdown("### 📊 Prediction Details")

            st.caption(
                "The following results show the model's "
                "five highest predicted class probabilities."
            )

            for probability, index in zip(
                top_values, top_indices
            ):
                label = model.config.id2label[index.item()]
                score = probability.item()

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**{label}**")
                    st.progress(score)

                with col2:
                    st.write(f"**{score * 100:.2f}%**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <h3 style="color:#d5f3f5;">🌊 AquaVision AI</h3>
    <p>Intelligent Fish Species Recognition</p>
    <p>Powered by Vision Transformer · PyTorch · Hugging Face · Streamlit</p>
</div>
""", unsafe_allow_html=True)
