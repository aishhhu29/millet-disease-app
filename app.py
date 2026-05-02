import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import hashlib
import cv2

# ===============================
# 🔐 LOGIN SYSTEM
# ===============================
def hash_pass(password):
    return hashlib.sha256(password.encode()).hexdigest()

users = {
    "admin": {"password": hash_pass("admin123"), "role": "admin"},
    "user": {"password": hash_pass("user123"), "role": "user"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.history = []

def login():
    st.markdown("<h2 style='text-align:center;'>🔐 Login</h2>", unsafe_allow_html=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in users and users[u]["password"] == hash_pass(p):
            st.session_state.logged_in = True
            st.session_state.role = users[u]["role"]
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ===============================
# 🎨 STYLE
# ===============================
st.markdown("""
<style>
.main-title {
    text-align:center;
    font-size:36px;
    color:#77dd77;
    font-weight:bold;
}
.card {
    background:white;
    padding:18px;
    border-radius:15px;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);
}
@media (max-width:768px){
    .main-title {font-size:24px;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌿 Digital Twin Millet System</div>", unsafe_allow_html=True)

# ===============================
# LOAD CLASS NAMES
# ===============================
class_names = np.load("class_names.npy", allow_pickle=True)
clean_names = [i.replace("_"," ").title() for i in class_names]

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
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)

# ===============================
# HEATMAP (FIXED)
# ===============================
def generate_heatmap(img):
    img_array = preprocess(img)

    last_conv = None
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            last_conv = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv, preds = grad_model(img_array)
        loss = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, conv)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    conv = conv[0]
    heatmap = conv @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = heatmap.numpy() if hasattr(heatmap, "numpy") else heatmap
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap, (224,224))

    img_np = np.array(img.resize((224,224)))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap*0.4 + img_np

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Analysis", "📄 Report"])

# ===============================
# HOME
# ===============================
with tab1:
    st.markdown("## 🌐 System Overview")

    col1, col2, col3 = st.columns(3)

    col1.markdown("### 🌱 Disease Detection")
    col1.write("Image-based classification")

    col2.markdown("### 🌡 Environment Simulation")
    col2.write("Digital Twin modeling")

    col3.markdown("### 📊 Severity Index")
    col3.write("Prediction + environment")

    st.markdown("---")
    st.info("👉 Go to Analysis tab to start")

# ===============================
# ANALYSIS
# ===============================
with tab2:
    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=250)

        pred = model.predict(preprocess(img))
        idx = np.argmax(pred)
        confidence = float(np.max(pred))
        disease = clean_names[idx]

        # METRICS
        col1, col2, col3 = st.columns(3)
        col1.metric("🌱 Disease", disease)
        col2.metric("📊 Confidence", f"{confidence*100:.2f}%")
        col3.metric("⚠ Severity", "High" if confidence > 0.7 else "Moderate")

        st.session_state.history.append((disease, confidence))

        # BAR CHART
        st.subheader("📊 Probabilities")
        fig = px.bar(x=clean_names, y=pred[0])
        st.plotly_chart(fig)

        # PIE CHART
        st.subheader("🥇 Top Predictions")
        top = np.argsort(pred[0])[-3:]
        fig2 = px.pie(values=pred[0][top], names=[clean_names[i] for i in top])
        st.plotly_chart(fig2)

        # HEATMAP
        if st.button("🔥 Show Heatmap"):
            st.image(generate_heatmap(img))

# ===============================
# REPORT
# ===============================
with tab3:
    st.subheader("🧠 Explainable Report")

    if st.session_state.history:
        last = st.session_state.history[-1]
        st.write(f"Disease: {last[0]}")
        st.write(f"Confidence: {last[1]*100:.2f}%")

    st.subheader("📊 History")
    for h in st.session_state.history:
        st.write(h)

# ===============================
# ADMIN
# ===============================
if st.session_state.role == "admin":
    st.sidebar.success("Admin Mode")
else:
    st.sidebar.info("User Mode")
