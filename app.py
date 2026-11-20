import html
import re

import streamlit as st
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

REPO_ID = "BismaLJ/fish_species_classification"

# Optional: your own fish vs non-fish model on Hugging Face (e.g. "BismaLJ/non_fish_classification").
# Leave "" to use the built-in zero-shot CLIP check instead.
NONFISH_REPO_ID = ""
FISH_THRESHOLD = 0.50      # min probability that the image is a fish
MIN_SPECIES_CONF = 0.35    # min confidence of the species model

st.set_page_config(page_title="FishVision AI", page_icon="🐟", layout="wide")

# ----------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }

.hero {
  background: linear-gradient(135deg, #062C43 0%, #0B5D75 55%, #14A3A8 100%);
  border-radius: 22px; padding: 2.4rem 2.6rem; color: #fff; margin-bottom: 1.6rem;
  box-shadow: 0 12px 30px rgba(6,44,67,.25); position: relative; overflow: hidden;
}
.hero:after { content: "🐟"; position: absolute; right: 2rem; bottom: -.6rem; font-size: 7rem; opacity: .15; }
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 700; letter-spacing: -.5px; color: #fff; }
.hero p { margin: .5rem 0 0; font-size: 1.02rem; opacity: .9; max-width: 640px; }
.badges { margin-top: 1.1rem; display: flex; gap: .6rem; flex-wrap: wrap; }
.badge { background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3);
  padding: .25rem .8rem; border-radius: 999px; font-size: .78rem; backdrop-filter: blur(4px); }

.card { background: #fff; border-radius: 18px; padding: 1.4rem 1.6rem;
  box-shadow: 0 4px 18px rgba(16,48,59,.08); border: 1px solid #E3EEF2; margin-bottom: 1rem; }
.card h3 { margin: 0 0 .8rem; font-size: 1.05rem; font-weight: 600; color: #0B5D75; }

.result-title { font-size: 1.7rem; font-weight: 700; color: #062C43; margin: .2rem 0; line-height: 1.2; }
.result-sub { color: #5B7C88; font-size: .9rem; margin-bottom: .9rem; }
.chip { display: inline-block; background: #D9F2EE; color: #0A6B62; font-weight: 600;
  padding: .3rem .9rem; border-radius: 999px; font-size: .85rem; }
.chip.warn { background: #FFF1D6; color: #8A5A00; }
.chip.bad { background: #FDE2E2; color: #A12B2B; }

.bar-row { margin: .65rem 0; }
.bar-head { display: flex; justify-content: space-between; font-size: .86rem; color: #10303B; margin-bottom: .25rem; }
.bar-bg { background: #E3EEF2; border-radius: 99px; height: 9px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #0E7C86, #3FC1C9); }

.alert { border-radius: 16px; padding: 1.3rem 1.5rem; margin-top: .4rem; }
.alert.bad { background: #FDECEC; border: 1px solid #F5C2C2; color: #7A1F1F; }
.alert.warn { background: #FFF6E3; border: 1px solid #F3DDA4; color: #70500A; }
.alert h4 { margin: 0 0 .3rem; font-size: 1.15rem; }
.alert p { margin: 0; font-size: .92rem; }

[data-testid="stFileUploader"] section { border: 2px dashed #7FC4CB; background: #F0FAFB; border-radius: 14px; }
[data-testid="stImage"] img { border-radius: 14px; }
section[data-testid="stSidebar"] { background: #062C43; }
section[data-testid="stSidebar"] * { color: #E8F4F7 !important; }
.footer { text-align: center; color: #7B96A0; font-size: .8rem; margin-top: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ models
@st.cache_resource(show_spinner="Loading species model...")
def load_model():
    processor = ViTImageProcessor.from_pretrained(REPO_ID)
    model = ViTForImageClassification.from_pretrained(REPO_ID)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading fish checker...")
def load_gate():
    if NONFISH_REPO_ID:
        return "own", ViTImageProcessor.from_pretrained(NONFISH_REPO_ID), \
            ViTForImageClassification.from_pretrained(NONFISH_REPO_ID).eval()
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    return "clip", CLIPProcessor.from_pretrained(name), CLIPModel.from_pretrained(name).eval()


def fish_probability(image: Image.Image) -> float:
    kind, proc, gate = load_gate()
    image = image.convert("RGB")
    with torch.no_grad():
        if kind == "own":
            probs = torch.softmax(gate(**proc(images=image, return_tensors="pt")).logits, -1)[0]
            p_non = sum(
                p.item() for i, p in enumerate(probs)
                if re.search(r"\bnon|\bnot\b|other", gate.config.id2label[i].lower())
            )
            return 1.0 - p_non
        prompts = [
            "a photo of a fish", "a photo of seafood", "a photo of a shrimp",
            "a photo of a squid", "a photo of a ray or shark",
            "a photo of a person", "a photo of an animal", "a photo of a building",
            "a photo of a vehicle", "a photo of food", "a photo of a plant",
            "a photo of an object", "a photo of a landscape", "a screenshot",
        ]
        out = gate(**proc(text=prompts, images=image, return_tensors="pt", padding=True))
        probs = out.logits_per_image.softmax(-1)[0]
        return probs[:5].sum().item()


def clean_label(raw: str):
    """'Balochi  gore \\nEnglish narrow-barred Spanish mackerel' -> (local, english)."""
    parts = re.split(r"\bEnglish\b", raw.strip(), maxsplit=1)
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


def esc(x: str) -> str:
    return html.escape(x)


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🐟 FishVision AI")
    st.markdown(
        "Identify **25 fish & seafood species** from the Balochistan coast using a "
        "fine-tuned **Vision Transformer (ViT)**."
    )
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Upload or capture a photo\n2. We verify it's a fish\n3. The AI predicts the species")
    st.markdown("---")
    st.markdown("**Tips**\n\n- Use a clear, well-lit photo\n- Keep the whole fish in frame\n- One fish per image")
    st.markdown("---")
    st.caption("Model: BismaLJ/fish_species_classification")

# -------------------------------------------------------------------- hero
st.markdown(
    """
<div class="hero">
  <h1>Fish Species Classifier</h1>
  <p>Upload a photo and get an instant AI-powered identification, with local and English names.</p>
  <div class="badges"><span class="badge">25 species</span><span class="badge">ViT-Base</span>
  <span class="badge">Non-fish detection</span></div>
</div>
""",
    unsafe_allow_html=True,
)

processor, model = load_model()

left, right = st.columns([1, 1.1], gap="large")

with left:
    st.markdown('<div class="card"><h3>📷 Your image</h3>', unsafe_allow_html=True)
    tab_up, tab_cam = st.tabs(["Upload", "Camera"])
    with tab_up:
        up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    with tab_cam:
        cam = st.camera_input("Take a picture", label_visibility="collapsed")
    file = up or cam
    if file is not None:
        image = Image.open(file)
        st.image(image, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    if file is None:
        st.markdown(
            '<div class="card"><h3>🔎 Result</h3><div class="result-sub">'
            "Your prediction will appear here once you add an image.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analyzing image..."):
            p_fish = fish_probability(image)
            results = predict(image, processor, model) if p_fish >= FISH_THRESHOLD else None

        if results is None:
            st.markdown(
                f'<div class="alert bad"><h4>🚫 Not a fish image</h4>'
                f"<p>This picture doesn't look like a fish (fish likelihood {p_fish*100:.0f}%). "
                "Please upload a clear photo of a fish.</p></div>",
                unsafe_allow_html=True,
            )
        elif results[0][1] < MIN_SPECIES_CONF:
            st.markdown(
                '<div class="alert warn"><h4>⚠️ Species not recognised</h4>'
                "<p>This looks like a fish, but it doesn't clearly match any of the 25 species "
                "this model knows. Try a clearer, closer photo.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            local, english = clean_label(results[0][0])
            conf = results[0][1]
            chip_cls = "" if conf >= 0.7 else "warn"
            bars = ""
            for label, p in results:
                loc, eng = clean_label(label)
                bars += (
                    f'<div class="bar-row"><div class="bar-head"><span>{esc(eng or loc)}</span>'
                    f"<span>{p*100:.1f}%</span></div>"
                    f'<div class="bar-bg"><div class="bar-fill" style="width:{p*100:.1f}%"></div></div></div>'
                )
            st.markdown(
                f"""
<div class="card"><h3>✅ Prediction</h3>
  <div class="result-title">{esc(english or local)}</div>
  <div class="result-sub">{('Local name: ' + esc(local)) if english else ''}</div>
  <span class="chip {chip_cls}">{conf*100:.1f}% confidence</span>
</div>
<div class="card"><h3>Top 5 matches</h3>{bars}</div>
""",
                unsafe_allow_html=True,
            )

st.markdown('<div class="footer">FishVision AI · Built with Streamlit & Hugging Face Transformers</div>',
            unsafe_allow_html=True)
