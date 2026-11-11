
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="wide"
)

# Model settings
MODEL_PATH = Path("facial_expression_model.keras")
CLASSES_PATH = Path("classes.json")
IMAGE_SIZE = (64, 64)

DEFAULT_CLASSES = [
    "angry", "disgusted", "fearful",
    "happy", "neutral", "sad", "surprised"
]

EMOJI = {
    "angry": "😠",
    "disgusted": "🤢",
    "fearful": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprised": "😮"
}

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

# Sidebar
with st.sidebar:
    st.title("😊 Emotion AI")
    page = st.radio(
        "Navigation",
        ["Overview", "Predict Expression", "About Model"]
    )
    st.divider()
    st.caption("CNN | TensorFlow | 64 × 64 RGB")

# Overview
if page == "Overview":

    st.title("😊 Facial Expression Recognition")

    st.write(
        "Upload a face image and use a trained CNN model "
        "to classify its facial expression."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "CNN")
    col2.metric("Image Size", "64 × 64")
    col3.metric("Classes", "7")

    st.subheader("Project Features")

    st.write("✔ Upload facial image")
    st.write("✔ Actual trained model prediction")
    st.write("✔ Seven expression classes")
    st.write("✔ Prediction score visualization")

# Prediction
elif page == "Predict Expression":

    st.title("📸 Upload Image for Prediction")

    uploaded_file = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            # Read uploaded image
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Image")
                st.image(
                    image,
                    caption="Your uploaded image",
                    use_container_width=True
                )

            with col2:
                st.subheader("Model Prediction")

                if st.button("🔍 Predict Expression",
                             type="primary"):

                    if not MODEL_PATH.exists():
                        st.error(
                            "Trained model not found! "
                            "Please add facial_expression_model.keras."
                        )
                        st.stop()

                    # Load actual trained model
                    with st.spinner("Analyzing image..."):

                        model = load_model()

                        # Load class names
                        if CLASSES_PATH.exists():
                            classes = json.loads(
                                CLASSES_PATH.read_text(
                                    encoding="utf-8"
                                )
                            )
                        else:
                            classes = DEFAULT_CLASSES

                        # Image preprocessing
                        resized_image = image.resize(IMAGE_SIZE)

                        img_array = np.asarray(
                            resized_image,
                            dtype=np.float32
                        )

                        # Add batch dimension
                        img_array = np.expand_dims(
                            img_array, axis=0
                        )

                        # Actual model prediction
                        predictions = model.predict(
                            img_array,
                            verbose=0
                        )[0]

                        # Highest prediction
                        predicted_index = int(
                            np.argmax(predictions)
                        )

                        predicted_class = classes[predicted_index]
                        confidence = float(
                            predictions[predicted_index]
                        )

                    st.success("Prediction Completed!")

                    st.markdown("### Predicted Expression")

                    st.markdown(
                        f"# {EMOJI.get(predicted_class, '🙂')} "
                        f"{predicted_class.title()}"
                    )

                    st.metric(
                        "Model Score",
                        f"{confidence * 100:.2f}%"
                    )

                    st.subheader("All Expression Scores")

                    for name, score in sorted(
                        zip(classes, predictions),
                        key=lambda x: x[1],
                        reverse=True
                    ):

                        st.write(
                            f"{EMOJI.get(name, '🙂')} "
                            f"**{name.title()}**"
                        )

                        st.progress(
                            float(np.clip(score, 0, 1))
                        )

                        st.caption(f"{score * 100:.2f}%")

                    st.info(
                        "This prediction is generated by the "
                        "trained CNN model. It is an educational "
                        "classification result, not a reliable "
                        "measurement of someone's actual emotions."
                    )

        except Exception as exc:
            st.error(f"Prediction Error: {exc}")

# About Model
else:

    st.title("About the CNN Model")

    st.write(
        "This project uses a Convolutional Neural Network "
        "(CNN) developed with TensorFlow and Keras."
    )

    st.write("**Input:** 64 × 64 RGB image")
    st.write("**Output:** Seven expression classes")

    st.write("**Expression Classes:**")

    for emotion in DEFAULT_CLASSES:
        st.write(
            f"{EMOJI[emotion]} {emotion.title()}"
        )

    st.warning(
        "This project is intended for educational purposes only."
    )
