import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===============================
# LOGIN SYSTEM
# ===============================
users = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.role = users[username]["role"]
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(layout="wide")

# ===============================
# TITLE
# ===============================
st.markdown("<h1 style='text-align:center; color:#77dd77;'>🌿 Digital Twin Millet System</h1>", unsafe_allow_html=True)

# ===============================
# LOAD DATA
# ===============================
class_names = np.load("class_names.npy", allow_pickle=True)
clean_names = [n.replace("_", " ").title() for n in class_names]

# ===============================
# MODEL
# ===============================
@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model as keras_load_model
    try:
        return keras_load_model("fixed_model.h5", compile=False)
    except:
        base = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224,224,3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        out = tf.keras.layers.Dense(len(clean_names), activation="softmax")(x)
        model = tf.keras.Model(base.input, out)
        model.load_weights("fixed_model.h5")
        return model

model = load_model()

# ===============================
# PREPROCESS
# ===============================
def preprocess(img):
    img = img.resize((224,224))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

# ===============================
# TABS (APP UI)
# ===============================
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Analysis", "📄 Report"])

# ===============================
# HOME
# ===============================
with tab1:
    st.subheader("🌐 Digital Twin Overview")
    st.write("System for disease detection and severity prediction using Digital Twin concept.")

# ===============================
# ANALYSIS
# ===============================
with tab2:
    uploaded = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

    col1, col2, col3 = st.columns(3)
    temp = col1.slider("Temperature", 10,50,25)
    humidity = col2.slider("Humidity",10,100,50)
    soil = col3.slider("Soil Moisture",10,100,50)

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=250)

        pred = model.predict(preprocess(img))
        idx = np.argmax(pred)
        confidence = float(np.max(pred))
        disease = clean_names[idx]

        st.metric("Disease", disease)
        st.metric("Confidence", f"{confidence*100:.2f}%")

        # Plotly chart
        fig = px.bar(
            x=clean_names,
            y=pred[0],
            title="Prediction Probabilities"
        )
        st.plotly_chart(fig)

# ===============================
# REPORT (Explainable Panel)
# ===============================
with tab3:
    st.subheader("🧠 Explainable Report")

    if "confidence" in locals():
        st.write(f"Detected Disease: **{disease}**")
        st.write(f"Confidence: **{confidence*100:.2f}%**")

        severity = (confidence*70)+(temp/50*10)+(humidity/100*10)+(soil/100*10)

        if severity > 70:
            level = "High"
        elif severity > 40:
            level = "Moderate"
        else:
            level = "Low"

        st.write(f"Severity Level: **{level}**")

        st.info("""
        The system analyzes leaf patterns and environmental conditions 
        to predict disease severity and provide recommendations.
        """)

# ===============================
# ADMIN PANEL
# ===============================
if st.session_state.role == "admin":
    st.sidebar.success("Admin Mode")
    st.sidebar.write("Additional controls can be added here")
else:
    st.sidebar.info("User Mode")
