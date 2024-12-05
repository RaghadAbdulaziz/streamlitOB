import streamlit as st
from PIL import Image
import numpy as np
import joblib
import cv2  # For image preprocessing if needed

# Load your SVM model
@st.cache_resource
def load_model():
    # Replace 'svm_model.pkl' with the actual path to your saved SVM model
    model = joblib.load("svm_model.pkl")
    return model

model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale (common for SVM tasks like classification)
    image = image.convert("L")  # Convert to grayscale
    image = np.array(image)

    # Resize to match the input size expected by your SVM model
    resized_image = cv2.resize(image, (64, 64))  # Example size; adjust as needed

    # Flatten the image to a 1D array (common for SVM input)
    flattened_image = resized_image.flatten()

    # Scale pixel values to [0, 1] range if needed
    normalized_image = flattened_image / 255.0
    return normalized_image

# Function to make predictions
def predict(image, model):
    # Preprocess the image
    processed_image = preprocess_image(image)
    processed_image = processed_image.reshape(1, -1)  # Reshape for scikit-learn

    # Make a prediction using the SVM model
    prediction = model.predict(processed_image)
    return prediction

# Streamlit App Interface
st.title("Image Classification with SVM")
st.write("Upload an image, and the SVM model will classify it.")

# Upload file
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform prediction
    st.write("Classifying the image...")
    prediction = predict(image, model)

    # Display the results
    st.write("Prediction:", prediction[0])


import streamlit as st

st.title("Test App")
st.write("Hello, Streamlit!")
