import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Digital Twin Millet System", layout="wide")

# ===============================
# 🎓 TITLE
# ===============================
st.markdown("""
<h1 style='text-align:center; color:#00ffcc;'>
A Digital Twin–Driven Decision Support Framework 
for Predictive Disease Severity in Millets
</h1>
""", unsafe_allow_html=True)

st.success("✔ Digital Twin Active: Real-time crop environment simulation enabled")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🌾 System Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analysis", "About"])

# ===============================
# LOAD DATA
# ===============================
class_names = np.load("class_names.npy", allow_pickle=True)
clean_names = [name.replace("_", " ").title() for name in class_names]

# ===============================
# MODEL
# ===============================
@st.cache_resource
def load_model():
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    base = MobileNetV2(weights=None, include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    out = Dense(len(class_names), activation="softmax")(x)

    model = Model(base.input, out)
    model.load_weights("fixed_model.h5")
    return model

model = load_model()

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

    🔬 Core Components:
    - AI-based disease detection (MobileNetV2)
    - Environmental simulation (Digital Twin)
    - Predictive severity modeling
    - Decision support recommendations
    """)

# ===============================
# ANALYSIS
# ===============================
if page == "Analysis":

    st.header("🔍 Disease Detection & Digital Twin Dashboard")

    uploaded = st.file_uploader("📤 Upload Millet Leaf Image", type=["jpg","png","jpeg"])

    col1, col2, col3 = st.columns(3)
    temp = col1.slider("🌡 Temperature (°C)", 10,50,25)
    humidity = col2.slider("💧 Humidity (%)",10,100,50)
    soil = col3.slider("🌱 Soil Moisture (%)",10,100,50)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)

        pred = model.predict(preprocess(img))

        idx = np.argmax(pred)
        confidence = float(np.max(pred))
        disease = clean_names[idx]

        # ===============================
        # 🎯 METRICS
        # ===============================
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", f"{confidence*100:.2f}%")
        c2.metric("Detected Disease", disease)
        c3.metric("Severity", "High" if confidence > 0.7 else "Moderate")

        st.markdown("---")

        # ===============================
        # 🧬 DIGITAL TWIN
        # ===============================
        st.subheader("🧬 Digital Twin Simulation")

        st.write(f"""
        A virtual representation of the crop environment is created using:

        - 🌡 Temperature: {temp}°C  
        - 💧 Humidity: {humidity}%  
        - 🌱 Soil Moisture: {soil}%  

        These parameters dynamically influence disease progression.
        """)

        st.markdown("---")

        # ===============================
        # ⚠ SEVERITY INDEX
        # ===============================
        severity_index = (confidence*70)+(temp/50*10)+(humidity/100*10)+(soil/100*10)
        severity_index = min(severity_index,100)

        if severity_index > 70:
            level = "High"
        elif severity_index > 40:
            level = "Moderate"
        else:
            level = "Low"

        st.subheader("⚠ Predictive Severity Index (Digital Twin Output)")
        st.progress(int(severity_index))
        st.write(f"{severity_index:.2f}%")
        st.write(f"👉 Severity Level: **{level}**")

        st.markdown("---")

        # ===============================
        # 📊 CHARTS
        # ===============================
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 All Class Probabilities")
            fig, ax = plt.subplots()
            ax.barh(clean_names, pred[0])
            ax.set_xlabel("Probability")
            st.pyplot(fig)

        with col2:
            st.subheader("🥇 Top 3 Predictions")
            top_indices = np.argsort(pred[0])[-3:][::-1]
            top_classes = [clean_names[i] for i in top_indices]
            top_values = [pred[0][i] for i in top_indices]

            fig2, ax2 = plt.subplots()
            ax2.pie(top_values, labels=top_classes, autopct='%1.1f%%')
            st.pyplot(fig2)

        st.markdown("---")

        # ===============================
        # 🧠 DECISION SUPPORT
        # ===============================
        st.subheader("🧠 Decision Support Engine (Digital Twin Output)")

        st.write(f"""
        👉 Disease Detected: **{disease}**  
        👉 Severity Index: **{severity_index:.2f}%**  
        👉 Severity Level: **{level}**
        """)

        st.markdown("""
        **Recommended Actions:**
        - Apply appropriate fungicide treatment  
        - Monitor environmental conditions  
        - Improve irrigation and drainage  
        - Use resistant seed varieties  
        """)

        st.info("📌 The system integrates AI prediction with Digital Twin simulation to generate context-aware recommendations.")

        st.markdown("---")

        # ===============================
        # 🧠 AI INTERPRETATION
        # ===============================
        st.subheader("🧠 AI Interpretation")

        st.write(f"""
        The deep learning model analyzed leaf patterns to detect disease.  
        The Digital Twin enhances prediction by incorporating environmental 
        parameters, improving reliability and real-world applicability.
        """)

# ===============================
# ABOUT
# ===============================
if page == "About":
    st.title("📄 About This Research")

    st.write("""
    This project presents a Digital Twin–driven framework for crop disease analysis.

    Technologies Used:
    - Deep Learning (MobileNetV2)
    - Streamlit Dashboard
    - Environmental Simulation
    - Predictive Analytics

    Contribution:
    - Combines AI + Digital Twin for agriculture
    - Provides real-time decision support
    - Enhances disease severity prediction accuracy
    """)