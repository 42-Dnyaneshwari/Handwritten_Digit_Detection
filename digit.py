import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained CNN model
model = tf.keras.models.load_model('mnist_cnn_model.h5')  # Make sure the model path is correct

# Function to preprocess uploaded image
def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for CNN input
    return img_array

# Streamlit App Interface
st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)  # Confidence score

        st.write(f"Predicted Digit: {predicted_label}")
        st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
