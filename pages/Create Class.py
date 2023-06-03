import streamlit as st
import os
from datetime import datetime
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

captured_image = None


def save_image(image, class_name):
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join("images", class_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{current_datetime}.jpg")
    with open(file_path, "wb") as file:
        file.write(image)


def transform_video_frame(frame):
    global captured_image
    # Convert the frame to an image
    img = frame.to_ndarray(format="bgr24")

    # Display the video frame
    st.image(img, channels="BGR")

    # Capture the image when a button is clicked
    if st.button("Capture Image"):
        captured_image = img

    return frame


st.title("Capture and Save Images")
class_name = st.text_input("Enter Class Name:")
if st.button("Capture"):
    image = st.camera_input("Capture a image")
    if image and class_name is not None:
        save_image(image, class_name)
        st.success("Image captured and saved successfully.")
    elif class_name is None:
        st.error("Insert a name")
