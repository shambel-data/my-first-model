import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Coffee Leaf Disease Classifier", layout="centered")
st.title("🍃 Coffee Leaf Disease Classifier")
st.write("Upload a picture of a coffee leaf to predict if it is healthy or affected by a disease.")

# Cache model loading to avoid reloading on every interface action
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('coffee_disease_model.h5')

try:
    model = load_my_model()
    class_names = ['Healthy', 'Miner', 'Phoma', 'Rust']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure 'coffee_disease_model.h5' is located in the same directory as this file.")

uploaded_file = st.file_uploader("Choose a leaf image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("Analyzing leaf pattern...")

    # Target sizing and reshaping matching MobileNetV2 input layer
    img_resized = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Inference run
    predictions = model.predict(img_array)
    scores = predictions[0]
    class_index = np.argmax(scores)
    label = class_names[class_index]
    confidence = scores[class_index] * 100

    if label == 'Healthy':
        st.success(f"Result: **{label}** ({confidence:.2f}% confidence)")
    else:
        st.warning(f"Result: **{label}** ({confidence:.2f}% confidence)")