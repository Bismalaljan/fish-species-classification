import io
import html
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoImageProcessor, ViTForImageClassification
from huggingface_hub import hf_hub_download
from PIL import Image
import streamlit.components.v1 as components

# ------------------------------------------------------------
# AquaVision AI / Fish Finder AI
# Two-stage pipeline:
# 1) ResNet18: Fish vs Non-fish
# 2) ViT: Fish species classification
# ------------------------------------------------------------

st.set_page_config(
    page_title="Fish Finder AI",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPECIES_MODEL_ID = "BismaLJ/fish_species_classification"
DETECTOR_REPO_ID = "BismaLJ/fish_nonfish_detector"
DETECTOR_FILENAME = "fish_nonfish_detector.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_models():
    # Load the original ViT species classifier.
    processor = AutoImageProcessor.from_pretrained(SPECIES_MODEL_ID)
    species_model = ViTForImageClassification.from_pretrained(SPECIES_MODEL_ID)
    species_model.to(DEVICE)
    species_model.eval()

    # Download and load the separate fish/non-fish detector.
    detector_path = hf_hub_download(
        repo_id=DETECTOR_REPO_ID,
        filename=DETECTOR_FILENAME,
    )
    checkpoint = torch.load(
        detector_path,
        map_location=DEVICE,
        weights_only=True,
    )

    detector = models.resnet18(weights=None)
    detector.fc = nn.Linear(detector.fc.in_features, 2)
    detector.load_state_dict(checkpoint["model_state_dict"])
    detector.to(DEVICE)
    detector.eval()

    class_names = checkpoint.get("class_names", ["Non-fish", "Fish"])
    if "threshold" not in checkpoint:
        raise ValueError(
            "The detector checkpoint has no calibrated threshold. "
            "Run the calibration-and-upload cell in your Colab notebook, "
            "then upload the updated fish_nonfish_detector.pth to Hugging Face."
        )
    threshold = 0.50

    detector_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return processor, species_model, detector, class_names, threshold, detector_transform


def detect_fish(image, detector, detector_transform, threshold):
    image_tensor = detector_transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = detector(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    fish_probability = float(probabilities[1].item())
    return fish_probability, fish_probability >= threshold


def identify_species(image, processor, species_model):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = species_model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[0]

    top_values, top_indices = torch.topk(probabilities, k=min(5, probabilities.numel()))
    predicted_id = int(top_indices[0].item())
    species = species_model.config.id2label.get(
        predicted_id, f"Class {predicted_id}"
    )
    confidence = float(top_values[0].item()) * 100

    top_results = []
    for score, index in zip(top_values.tolist(), top_indices.tolist()):
        label = species_model.config.id2label.get(index, f"Class {index}")
        top_results.append((label, score * 100))

    return species, confidence, top_results


def speech_controls(message, key):
    safe_message = html.escape(message, quote=True)
    components.html(
        f"""
        <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;
                    font-family:Arial,sans-serif;">
          <button id="speak-{key}" style="background:#087f8c;color:white;
            border:0;border-radius:24px;padding:10px 18px;font-weight:600;
            cursor:pointer;">🔊 Speak result</button>
          <button id="stop-{key}" style="background:#e8eef0;color:#183b40;
            border:0;border-radius:24px;padding:10px 18px;font-weight:600;
            cursor:pointer;">Stop</button>
        </div>
        <script>
          const message = "{safe_message}";
          document.getElementById("speak-{key}").onclick = () => {{
            if (!("speechSynthesis" in window)) {{
              alert("Speech is not supported by this browser.");
              return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.rate = 0.92;
            window.speechSynthesis.speak(utterance);
          }};
          document.getElementById("stop-{key}").onclick = () => {{
            if ("speechSynthesis" in window) window.speechSynthesis.cancel();
          }};
        </script>
        """,
        height=58,
    )


# ------------------------------ Styling ------------------------------
st.markdown("""
<style>
.stApp { background: #f4f8f8; color: #17383c; }
header[data-testid="stHeader"] { background: #0c454b; }
.topbar {
    background: linear-gradient(100deg,#073f46,#11636a);
    color: white; padding: 18px 24px; border-radius: 0 0 16px 16px;
    margin: -5.5rem -1rem 1.4rem -1rem;
    font-size: 27px; font-weight: 750; letter-spacing: .2px;
}
.eyebrow { color:#16858a; font-weight:700; letter-spacing:2px;
           font-size:12px; text-transform:uppercase; }
.main-title { color:#153d42; font-size:32px; font-weight:800;
              margin:4px 0 18px 0; }
.panel {
    background:white; border:1px solid #e1ebeb; border-radius:18px;
    padding:20px; box-shadow:0 7px 24px rgba(15,65,70,.06);
}
.result {
    background:#e3f6e9; border:1px solid #b9e5c6; color:#145b35;
    border-radius:15px; padding:18px 20px; margin-top:18px;
}
.reject {
    background:#fff0ed; border:1px solid #ffd0c8; color:#8b3024;
    border-radius:15px; padding:18px 20px; margin-top:18px;
}
div.stButton > button {
    border-radius:24px; min-height:44px; font-weight:700;
}
[data-testid="stSidebar"] { background:#ffffff; }
.small-muted { color:#718589; font-size:13px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="topbar">🐟 &nbsp; Fish Finder AI</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🐠 Fish Finder AI")
    st.caption("Your fish identification assistant")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Fish Identification", "My Identifications", "Explore Species", "Settings"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<p class="small-muted">AquaVision AI · Powered by ResNet18 + ViT</p>',
                unsafe_allow_html=True)

if page == "Fish Identification":
    st.markdown('<div class="eyebrow">AquaVision AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Fish Identification</div>', unsafe_allow_html=True)
    st.write("Upload a fish photo or capture one with your camera to identify its species.")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Choose an image")
        uploaded_file = st.file_uploader(
            "Select from Gallery",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        camera_file = st.camera_input("Capture New Photo")
        st.markdown("</div>", unsafe_allow_html=True)

    image_file = camera_file if camera_file is not None else uploaded_file

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Image preview")
        if image_file is not None:
            try:
                image = Image.open(image_file).convert("RGB")
                st.image(image, use_container_width=True)
            except Exception:
                st.error("This image could not be opened. Please choose another image.")
                image = None
        else:
            image = None
            st.info("Your selected image will appear here.")
        st.markdown("</div>", unsafe_allow_html=True)

    if image is not None:
        if st.button("Identify Fish", type="primary", use_container_width=True):
            try:
                with st.spinner("Checking the image and identifying the species..."):
                    processor, species_model, detector, class_names, threshold, detector_transform = load_models()
                    fish_probability, is_fish = detect_fish(
                        image, detector, detector_transform, threshold
                    )

                    if not is_fish:
                        st.markdown(
                            f"""
                            <div class="reject">
                              <h3>✕ This is not a fish image</h3>
                              <p>The fish detector's score was {fish_probability*100:.1f}%.
                              Please upload a clear fish photograph.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        speech_controls("This is not a fish image. Please upload a fish photograph.", "reject")
                    else:
                        species, confidence, top_results = identify_species(
                            image, processor, species_model
                        )
                        st.markdown(
                            f"""
                            <div class="result">
                              <h3>✓ Prediction: {html.escape(species)}</h3>
                              <h4>Confidence: {confidence:.1f}%</h4>
                              <p>Fish detector score: {fish_probability*100:.1f}%</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("### Top 5 predictions")
                        for label, score in top_results:
                            st.write(f"**{label}** — {score:.1f}%")
                            st.progress(min(max(score / 100, 0.0), 1.0))
                        speech_controls(
                            f"The detected fish species is {species}. Confidence {confidence:.1f} percent.",
                            "fish",
                        )
            except Exception as error:
                st.error("The models could not be loaded or the prediction failed.")
                st.exception(error)

elif page == "My Identifications":
    st.markdown('<div class="main-title">My Identifications</div>', unsafe_allow_html=True)
    st.info("Identification history is not saved yet. This can be added as a future feature.")

elif page == "Explore Species":
    st.markdown('<div class="main-title">Explore Species</div>', unsafe_allow_html=True)
    st.write("The species classifier recognizes the 15 fish classes included in its training dataset.")
    st.caption("Species names and predictions are provided by the trained ViT model.")

else:
    st.markdown('<div class="main-title">Settings</div>', unsafe_allow_html=True)
    st.write("Device:", "GPU" if torch.cuda.is_available() else "CPU")
    st.caption("Speech uses your browser or device's built-in text-to-speech voice.")
    st.caption("The fish detector was trained on underwater fish and empty-scene images; performance on unrelated everyday objects has not been fully validated.")
