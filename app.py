import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO

# Load YOLOv8 model

model = YOLO('./best.pt')  # Replace with the path to your trained YOLOv8 model

# App title and description
st.set_page_config(page_title="Plant Detection App", layout="wide")
st.title("🌱 Plant Species Detection App")
st.write("Upload an image to detect the plant species using a YOLOv8 model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Run YOLOv8 model on the uploaded image
#     with st.spinner("Detecting..."):
#         results = model(image)

#     # Display results
#     st.subheader("Detection Results")
#     results.render()  # Annotates the image
#     annotated_image = results.ims[0]

#     # Show the detected image with bounding boxes
#     st.image(annotated_image, caption="Detected Image", use_column_width=True)

#     # Display detected class names and confidence scores
#     for detection in results.xyxy[0]:  # xyxy format for each detection
#         class_id = int(detection[5])
#         class_name = model.names[class_id]
#         confidence = detection[4]
#         st.write(f"Detected: {class_name} with confidence {confidence:.2f}")
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv8 model on the uploaded image
    with st.spinner("Detecting..."):
        results = model(image)

    # Display results
    st.subheader("Detection Results")
    
    # Use the plot() method to get the annotated image
    annotated_image = results[0].plot()  # Access the first result and plot the detections

    # Show the detected image with bounding boxes
    st.image(annotated_image, caption="Detected Image", use_column_width=True)

    # Display detected class names and confidence scores
    for detection in results[0].boxes:  # Use .boxes to access detection boxes
        class_id = int(detection.cls[0])
        class_name = model.names[class_id]
        confidence = detection.conf[0]
        st.write(f"Detected: {class_name} with confidence {confidence:.2f}")








# Sidebar for extra functionalities
st.sidebar.title("About")
st.sidebar.info(
    """
    This app detects plant species using a YOLOv8 model trained on four classes:
    1. Azadiractha Indica
    2. Calotropis
    3. Ficus Religiosa (Raavi)
    4. Oleander

    Developed by Mudassir Fayaz.
    """
)

st.sidebar.title("How it works")
st.sidebar.write(
    """
    1. Upload an image of a plant.
    2. The app will process the image and identify the plant species.
    3. Results will be displayed with bounding boxes and confidence scores.
    """
)
