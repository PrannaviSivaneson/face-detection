import cv2
import numpy as np
import streamlit as st
from keras.models import model_from_json

# Load the model architecture from JSON file
with open("model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("model_weights.h5")

# Load the class labels
class_labels = np.load("class_labels.npy", allow_pickle=True).item()


# Function to preprocess an image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (image_width, image_height))
    image = image.astype("float32") / 255.0
    return image


# Function to predict the class of an image
def predict_image_class(image):
    preprocessed_image = preprocess_image(image)
    input_image = np.expand_dims(preprocessed_image, axis=0)
    predictions = loaded_model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[str(predicted_class_index)]
    return predicted_class_label


# Set the dimensions for the captured image
image_width = 224
image_height = 224

# Streamlit web application
st.title("Image Classification")
st.write("Click the button below to capture an image")

# Create a placeholder for the captured image
captured_image = st.empty()

img_file_buffer = st.camera_input("Take a picture")
if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    # st.image(cv2_img)
    try:
        predicted_class = predict_image_class(cv2_img)
        st.write("Predicted class:", predicted_class)

    except Exception as err:
        st.write(err)
        st.stop()

#
