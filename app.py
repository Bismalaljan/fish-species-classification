import re

import streamlit as st
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

REPO_ID = "BismaLJ/fish_species_classification"

st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    processor = ViTImageProcessor.from_pretrained(REPO_ID)
    model = ViTForImageClassification.from_pretrained(REPO_ID)
    model.eval()
    return processor, model


def clean_label(raw: str):
    """Labels look like 'Balochi  gore \\nEnglish narrow-barred Spanish mackerel'.
    Return (local_name, english_name)."""
    raw = raw.strip()
    parts = re.split(r"\bEnglish\b", raw, maxsplit=1)
    local = re.sub(r"\s+", " ", parts[0]).strip(" :")
    english = re.sub(r"\s+", " ", parts[1]).strip(" :") if len(parts) > 1 else ""
    return local, english


def predict(image: Image.Image, processor, model, top_k: int = 5):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top = torch.topk(probs, k=min(top_k, probs.numel()))
    return [(model.config.id2label[i.item()], p.item()) for p, i in zip(top.values, top.indices)]


st.title("🐟 Fish Species Classifier")
st.write(
    "Upload a photo of a fish and the fine-tuned Vision Transformer (ViT) will "
    "predict which of the 25 species it is."
)

processor, model = load_model()

source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
if source == "Upload":
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input("Take a picture")

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Your image", use_container_width=True)

    with st.spinner("Classifying..."):
        results = predict(image, processor, model)

    local, english = clean_label(results[0][0])
    st.success(f"**Prediction:** {english or local}  ({results[0][1]*100:.1f}% confidence)")
    if english:
        st.caption(f"Local name: {local}")

    st.subheader("Top 5 predictions")
    for label, p in results:
        loc, eng = clean_label(label)
        st.write(f"{eng or loc} — {p*100:.1f}%")
        st.progress(min(max(p, 0.0), 1.0))
     





          
    )




