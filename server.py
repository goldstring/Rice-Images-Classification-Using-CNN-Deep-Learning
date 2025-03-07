import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('model.h5')

# Define class names (adjust based on your dataset)
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Function to preprocess the image
def preprocess_image(image, target_size=(255, 255)):  # Adjust size if different
    image = image.resize(target_size)  # Resize the image
    image_array = img_to_array(image)  # Convert to array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize the image
    return image_array

# Streamlit app title
st.title("Rice Variety Classification")
st.write("Upload an image of rice grains to predict its variety.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image",width=200)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make a prediction
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]

    # Display the result
    st.write(f"### Predicted Variety: {predicted_class}")
    st.write(f"### Prediction Confidence: {np.max(predictions) * 100:.2f}%")
