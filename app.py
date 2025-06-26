import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = 256
MODEL_PATH = 'LungCancerPrediction.h5'
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

model = tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")
st.title("Lung Cancer Detection from Chest X-ray")
st.markdown("Upload a chest X-ray image to predict the type of lung condition.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

    if st.button("Predict"):
        try:
            input_data = preprocess_image(image)
            predictions = model.predict(input_data)
            predicted_index = np.argmax(predictions[0])
            predicted_label = CLASS_NAMES[predicted_index]

            st.subheader(f"Prediction: {predicted_label}")

        except Exception as e:
            st.error("Prediction failed.")
