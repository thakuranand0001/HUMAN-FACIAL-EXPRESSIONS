
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

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

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

st.title("😊 Facial Expression Prediction")

uploaded_file = st.file_uploader(
    "Upload a Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict Expression", type="primary"):

        if not MODEL_PATH.exists():
            st.error("Trained model not found!")
            st.stop()

        try:
            model = load_model()

            if CLASSES_PATH.exists():
                classes = json.loads(
                    CLASSES_PATH.read_text(encoding="utf-8")
                )
            else:
                classes = DEFAULT_CLASSES

            # Image preprocessing
            image = image.resize(IMAGE_SIZE)

            img_array = np.asarray(
                image, dtype=np.float32
            )

            img_array = np.expand_dims(
                img_array, axis=0
            )

            # Actual CNN model prediction
            predictions = model.predict(
                img_array, verbose=0
            )[0]

            # Get predicted expression
            index = np.argmax(predictions)
            predicted_expression = classes[index]
            confidence = float(predictions[index])

            # Display result
            st.success("Prediction Completed!")

            st.subheader("Predicted Expression")

            st.markdown(
                f"# {EMOJI.get(predicted_expression, '🙂')} "
                f"{predicted_expression.title()}"
            )

            st.metric(
                "Model Confidence Score",
                f"{confidence * 100:.2f}%"
            )

            # Display all expression scores
            st.subheader("Expression Scores")

            for name, score in sorted(
                zip(classes, predictions),
                key=lambda x: x[1],
                reverse=True
            ):
                st.write(
                    f"{EMOJI.get(name, '🙂')} **{name.title()}**"
                )
                st.progress(float(np.clip(score, 0, 1)))
                st.write(f"{score * 100:.2f}%")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
