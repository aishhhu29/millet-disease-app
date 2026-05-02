import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Millet Disease Predictor",
    page_icon="🌿",
    layout="centered"
)

# ------------------ CSS ------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 38px;
    color: #2e7d32;
    font-weight: bold;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #555;
}
.result-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL (FIXED) ------------------
@st.cache_resource
def load_my_model():
    return load_model("fixed_model.h5", compile=False, safe_mode=False)

model = load_my_model()

# Load class names safely
try:
    class_names = np.load("class_names.npy", allow_pickle=True)
except:
    class_names = ["Class 1", "Class 2", "Class 3"]

# ------------------ HEADER ------------------
st.markdown("<div class='main-title'>🌿 Millet Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload a millet leaf image to detect disease</div>", unsafe_allow_html=True)

st.write("---")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📂 Upload Leaf Image", type=["jpg", "jpeg", "png"])

# ------------------ PROCESS ------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # ------------------ RESULT ------------------
        st.write("### 🧾 Prediction Result")
        st.markdown(f"""
        <div class='result-box'>
            <h3>🌱 Disease: <span style='color:#d32f2f'>{class_names[index]}</span></h3>
            <h4>📊 Confidence: <span style='color:#1976d2'>{confidence*100:.2f}%</span></h4>
        </div>
        """, unsafe_allow_html=True)

        # ------------------ CHART ------------------
        st.write("### 📊 Prediction Confidence")

        fig, ax = plt.subplots()
        ax.bar(class_names, prediction[0])
        plt.xticks(rotation=45)
        plt.ylabel("Confidence")

        st.pyplot(fig)

    except Exception as e:
        st.error("⚠️ Error processing image. Please upload a valid image.")

# ------------------ FOOTER ------------------
st.write("---")
st.markdown(
    "<p style='text-align:center;'>AI-Based Millet Disease Detection | Streamlit App 🌾</p>",
    unsafe_allow_html=True
)
