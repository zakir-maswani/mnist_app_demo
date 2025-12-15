import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Page config
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✍️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_final_99.keras")

model = load_model()

st.title("✍️ Handwritten Digit Classifier")
st.write("Draw a digit (0–9) and let the CNN predict it")

st.divider()

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convert canvas image to PIL
    img = Image.fromarray(canvas_result.image_data.astype("uint8"))
    img = img.convert("L")        # grayscale
    img = img.resize((28, 28))    # MNIST size

    img_array = np.array(img)

    # Normalize (already white on black, no invert needed)
    img_array = img_array / 255.0
    img_array = img_array.astype("float32")
    img_array = img_array.reshape(1, 28, 28, 1)

    st.subheader("Model Input (28×28)")
    st.image(img_array.reshape(28,28), width=150)

    # Predict
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {digit}")
    st.info(f"Confidence: {confidence:.2f}%")
