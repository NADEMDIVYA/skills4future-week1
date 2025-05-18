import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Define class names (taken from your notebook, order is important!)
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

# Set Streamlit title
st.title("🌿 Plant Disease Detection System")
st.write("Upload a plant leaf image to detect the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Prediction function
def predict(image):
    image = image.resize((224, 224))  # Resize to model input size
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)

    # Debugging info
    
    predicted_index = np.argmax(predictions)

    if predicted_index >= len(class_names):
        st.error(f"Predicted index {predicted_index} exceeds class_names length {len(class_names)}.")
        st.stop()

    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions))
    return predicted_class, confidence

# Display image and prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        label, confidence = predict(image)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence * 100:.2f}%")
       
