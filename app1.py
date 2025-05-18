import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Google Drive model file ID
file_id = "1LdF7a5f8Tf2U59PpPn3-7D5zsAeodPeq"  # <- Replace with your real file ID
model_path = "Plant_disease_model.keras"

# Download and load model
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()

# class_names = [...]  # Make sure this has 38 items!
class_names = [
   'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy'
]

def predict(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)

    if predicted_index >= len(class_names):
        st.error(f"Predicted index {predicted_index} exceeds class_names length.")
        st.stop()

    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions))
    return predicted_class, confidence

st.title("🌿Plant Disease Detection System")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        label, confidence = predict(image)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence * 100:.2f}%")
