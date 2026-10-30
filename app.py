"""
Streamlit Web Application for COVID-19 CT Scan Classification
Author: AI Coding Assistant

Features:
- Educational web interface for 3-class CNN CT image inference
- Upload support for JPG, JPEG, PNG
- Preprocessing matches training pipeline (128x128 RGB, [0, 1] normalization)
- Displays predicted class, confidence scores, and probability bar chart
- Clear medical disclaimer and graceful model-missing error handling
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="COVID-19 CT Scan Classification",
    page_icon="🫁",
    layout="centered"
)

# Constants
MODEL_PATH = "models/covid_ct_cnn.keras"
MAPPING_PATH = "models/class_mapping.json"
IMAGE_SIZE = (128, 128)

def load_model_and_mapping():
    """Load the trained Keras model and class name mapping."""
    if not os.path.exists(MODEL_PATH):
        return None, None
        
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load mapping if exists, else fallback to standard 3 classes
        if os.path.exists(MAPPING_PATH):
            with open(MAPPING_PATH, "r") as f:
                mapping = json.load(f)
                class_names = mapping.get("class_names", ["COVID Positive", "Negative/Normal", "Non-Informative"])
        else:
            class_names = ["COVID Positive", "Negative/Normal", "Non-Informative"]
            
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(pil_img):
    """
    Apply identical preprocessing to training pipeline:
    - Convert to RGB (3 channels)
    - Resize to 128x128
    - Normalize pixel values to 0.0 - 1.0
    - Add batch dimension -> (1, 128, 128, 3)
    """
    rgb_img = pil_img.convert("RGB")
    resized_img = rgb_img.resize(IMAGE_SIZE)
    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    batch_img = np.expand_dims(img_array, axis=0)
    return batch_img

def main():
    # Header
    st.title("COVID-19 CT Scan Classification")
    st.caption("Three-Class CNN Image Classification")

    # Introduction Section
    st.markdown("""
    ### About this Project
    This application utilizes a Convolutional Neural Network (CNN) trained on preprocessed lung Computed Tomography (CT) scans to classify images into one of three categories:
    
    1. **COVID Positive:** CT scans displaying pulmonary patterns and lesions associated with COVID-19 pneumonia.
    2. **Normal:** CT scans showing normal lungs or findings unrelated to COVID-19.
    3. **Non-Informative:** Scans lacking sufficient pulmonary parenchyma structure (e.g. neck region, bone views, or blurry slices) necessary for assessment.
    """)

    st.markdown("---")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        st.warning("""
        ⚠️ **Model not found!**
        
        The trained CNN model (`models/covid_ct_cnn.keras`) is not available yet.
        Please run the training pipeline first:
        ```bash
        python train.py
        ```
        Once training completes, refresh this page to make predictions.
        """)
        return

    # Load Model
    model, class_names = load_model_and_mapping()
    if model is None:
        st.error("Failed to load model. Please verify `models/covid_ct_cnn.keras`.")
        return

    # Image Upload Section
    st.subheader("Upload CT Scan Image")
    uploaded_file = st.file_uploader(
        "Choose a CT scan image (JPG, JPEG, or PNG format):",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded CT Scan", use_column_width=False, width=320)
        except Exception as e:
            st.error(f"Invalid image file: {e}")
            return

        # Predict Button
        if st.button("Predict Classification", type="primary"):
            with st.spinner("Analyzing CT scan image..."):
                processed = preprocess_image(image)
                probabilities = model.predict(processed)[0]
                pred_index = int(np.argmax(probabilities))
                predicted_class = class_names[pred_index]
                confidence = probabilities[pred_index] * 100.0

            st.markdown("---")
            st.subheader("Prediction Results")

            # Result highlight
            if predicted_class == "COVID Positive":
                st.error(f"**Predicted Class:** {predicted_class}")
            elif predicted_class == "Normal":
                st.success(f"**Predicted Class:** {predicted_class}")
            else:
                st.info(f"**Predicted Class:** {predicted_class}")

            st.write(f"**Highest Model Confidence Score:** `{confidence:.2f}%`")

            # Probability Breakdown
            st.write("#### Output Confidence Scores by Class:")
            
            # Format dataframe / chart
            chart_data = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
            st.bar_chart(chart_data)

            for i, cname in enumerate(class_names):
                st.write(f"- **{cname}:** `{probabilities[i] * 100:.2f}%`")

    # Disclaimer Section
    st.markdown("---")
    st.info("""
    **Medical Disclaimer:**
    This project is for educational and research purposes only. It is not a medical diagnostic tool. Predictions may be incorrect and must not be used to diagnose, exclude, or treat COVID-19 or any other medical condition.
    """)

if __name__ == "__main__":
    main()
