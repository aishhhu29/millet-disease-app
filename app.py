import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===============================
# ⚙️ PAGE CONFIG
# ===============================
st.set_page_config(page_title="Digital Twin Millet System", layout="wide")

# ===============================
# 🎨 UI STYLE
# ===============================
st.markdown("""
<style>
.main-title {
    text-align:center;
    font-size:40px;
    color:#00c853;
    font-weight:bold;
}
.card {
    background:#ffffff;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 🎓 TITLE
# ===============================
st.markdown("<div class='main-title'>🌿 Digital Twin Millet Disease System</div>", unsafe_allow_html=True)
st.success("✔ AI + Digital Twin Active")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🌾 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analysis", "About"])

# ===============================
# LOAD CLASS NAMES
# ===============================
try:
    class_names = np.load("class_names.npy", allow_pickle=True)
    clean_names = [name.replace("_", " ").title() for name in class_names]
except:
    clean_names = ["Disease A", "Disease B", "Healthy"]

# ===============================
# MODEL LOADING (FIXED)
# ===============================
@st.cache_resource
def load_my_model():
    from tensorflow.keras.models import load_model as keras_load_model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    try:
        # Try loading full model
        return keras_load_model("fixed_model.h5", compile=False)
    except:
        # Fallback: rebuild model + load weights
        base = MobileNetV2(weights=None, include_top=False, input_shape=(224,224,3))
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(128, activation="relu")(x)
        out = Dense(len(clean_names), activation="softmax")(x)

        model = Model(base.input, out)
        model.load_weights("fixed_model.h5")
        return model

model = load_my_model()

# ===============================
# PREPROCESS
# ===============================
def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

# ===============================
# HOME
# ===============================
if page == "Home":
    st.header("🌐 Digital Twin Overview")
    st.write("""
    This system integrates Deep Learning with Digital Twin technology 
    to simulate crop conditions and predict disease severity in millets.
    """)

# ===============================
# ANALYSIS
# ===============================
if page == "Analysis":

    st.header("🔍 AI + Digital Twin Dashboard")

    uploaded = st.file_uploader("📤 Upload Millet Leaf Image", type=["jpg","png","jpeg"])

    col1, col2, col3 = st.columns(3)
    temp = col1.slider("🌡 Temperature (°C)", 10,50,25)
    humidity = col2.slider("💧 Humidity (%)",10,100,50)
    soil = col3.slider("🌱 Soil Moisture (%)",10,100,50)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        c1, c2 = st.columns([1,2])

        with c1:
            st.image(img, caption="Uploaded Image")

        pred = model.predict(preprocess(img))
        idx = np.argmax(pred)
        confidence = float(np.max(pred))
        disease = clean_names[idx]

        # ===============================
        # 🎯 METRICS
        # ===============================
        with c2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{confidence*100:.2f}%")
            m2.metric("Disease", disease)
            m3.metric("Severity", "High" if confidence > 0.7 else "Moderate")

        st.markdown("---")

        # ===============================
        # DIGITAL TWIN
        # ===============================
        st.subheader("🧬 Digital Twin Simulation")

        severity_index = (confidence*70)+(temp/50*10)+(humidity/100*10)+(soil/100*10)
        severity_index = min(severity_index,100)

        if severity_index > 70:
            level = "High"
        elif severity_index > 40:
            level = "Moderate"
        else:
            level = "Low"

        st.progress(int(severity_index))
        st.write(f"Severity Index: {severity_index:.2f}%")
        st.write(f"Level: **{level}**")

        st.markdown("---")

        # ===============================
        # CHARTS
        # ===============================
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Class Probabilities")
            fig, ax = plt.subplots()
            ax.barh(clean_names, pred[0])
            st.pyplot(fig)

        with col2:
            st.subheader("🥇 Top Predictions")
            top_indices = np.argsort(pred[0])[-3:][::-1]
            top_classes = [clean_names[i] for i in top_indices]
            top_values = [pred[0][i] for i in top_indices]

            fig2, ax2 = plt.subplots()
            ax2.pie(top_values, labels=top_classes, autopct='%1.1f%%')
            st.pyplot(fig2)

        st.markdown("---")

        # ===============================
        # DECISION SUPPORT
        # ===============================
        st.subheader("🧠 Decision Support")

        st.info(f"""
        Disease: {disease}  
        Severity: {level}  
        Recommended Actions:
        - Apply fungicide
        - Monitor environment
        - Improve irrigation
        """)

# ===============================
# ABOUT
# ===============================
if page == "About":
    st.title("📄 About Project")
    st.write("""
    AI + Digital Twin system for millet disease prediction.
    """)
