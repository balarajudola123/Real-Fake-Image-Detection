import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# -----------------------------
# Get project directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
MODEL_PATH = os.path.join(BASE_DIR, "real_fake_cnn_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "real_fake_classes.npy")

# -----------------------------
# Load model and class labels
# -----------------------------
@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()

    if not os.path.exists(CLASSES_PATH):
        st.error(f"❌ Class labels file not found: {CLASSES_PATH}")
        st.stop()

    model = load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    return model, classes


model, classes = load_model_and_classes()

# -----------------------------
# Image Classification Function
# -----------------------------
def classify_image(image):
    try:
        img = image.convert("RGB")
        img = img.resize((64, 64))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        class_index = np.argmax(prediction)
        confidence = float(prediction[0][class_index])

        return str(classes[class_index]), confidence

    except Exception as e:
        return None, str(e)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Real vs Fake Image Detection",
    page_icon="🖼️",
    layout="centered"
)

st.title("🧠 Real vs Fake Image Detection")
st.write("Upload an image and the model will predict whether it is **Real** or **Fake**.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Analyzing image..."):

        label, confidence = classify_image(image)

    if label is not None:
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence:.2%}**")
    else:
        st.error(label)

else:
    st.info("📁 Please upload an image.")
