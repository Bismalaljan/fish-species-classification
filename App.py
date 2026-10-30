import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch.nn.functional as F


# --------------------------------
# 1. Application configuration
# --------------------------------

st.set_page_config(
    page_title="Fish Species Classification",
    page_icon="🐟",
    layout="centered"
)

st.title("🐟 Fish Species Classification")
st.write(
    "Upload or capture a fish image to predict its species "
    "using a fine-tuned Vision Transformer model."
)

st.divider()


# --------------------------------
# 2. Hugging Face model details
# --------------------------------

MODEL_ID = "BismaLJ/fish_species_classification"


# --------------------------------
# 3. Select CPU or GPU
# --------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------
# 4. Load model and processor
# --------------------------------

@st.cache_resource
def load_model():

    processor = ViTImageProcessor.from_pretrained(
        MODEL_ID
    )

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID
    )

    model.to(device)
    model.eval()

    return model, processor


with st.spinner("Loading fish classification model..."):

    try:
        model, processor = load_model()
        st.success("Model loaded successfully!")

    except Exception as e:
        st.error(f"Could not load the model: {e}")
        st.stop()


# --------------------------------
# 5. Choose image input method
# --------------------------------

st.subheader("Choose an Image")

input_method = st.radio(
    "How would you like to provide an image?",
    ["Upload Image", "Capture with Camera"],
    horizontal=True
)

image_file = None

if input_method == "Upload Image":

    image_file = st.file_uploader(
        "Upload a fish image",
        type=["jpg", "jpeg", "png"]
    )

else:

    image_file = st.camera_input(
        "Take a picture of a fish"
    )


# --------------------------------
# 6. Image prediction function
# --------------------------------

def predict_fish(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Preprocess image using ViT processor
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Move input tensors to the same device as model
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Make prediction
    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = F.softmax(
            outputs.logits,
            dim=-1
        )

    # Get top five predictions
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=min(5, probabilities.shape[-1]),
        dim=-1
    )

    results = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        species = model.config.id2label[
            index.item()
        ].strip()

        confidence = probability.item() * 100

        results.append(
            (species, confidence)
        )

    return results


# --------------------------------
# 7. Display image and prediction
# --------------------------------

if image_file is not None:

    image = Image.open(image_file).convert("RGB")

    st.subheader("Selected Fish Image")

    st.image(
        image,
        caption="Input Image",
        use_container_width=True
    )

    if st.button(
        "Classify Fish",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            try:

                predictions = predict_fish(image)

                st.divider()

                st.subheader("Prediction Result")

                best_species, best_confidence = predictions[0]

                st.success(
                    f"Predicted Species: {best_species}"
                )

                st.metric(
                    label="Prediction Confidence",
                    value=f"{best_confidence:.2f}%"
                )

                st.divider()

                st.subheader("Top 5 Predictions")

                for species, confidence in predictions:

                    st.write(
                        f"**{species} — {confidence:.2f}%**"
                    )

                    st.progress(
                        min(confidence / 100, 1.0)
                    )

                st.info(
                    "Note: The model always selects from its "
                    "15 trained fish categories. A high confidence "
                    "score does not guarantee that the prediction "
                    "is correct."
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# --------------------------------
# 8. Footer
# --------------------------------

st.divider()

st.caption(
    "Fish Species Classification | "
    "Vision Transformer (ViT) | "
    "BS Information Technology Project"
)

st.caption(
    "Model hosted on Hugging Face: "
    "BismaLJ/fish_species_classification"
)
