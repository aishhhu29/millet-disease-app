import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import hashlib
import cv2

# ===============================
# 🔐 LOGIN
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
# 🎨 UI STYLE
# ===============================
st.markdown("""
<style>
.main-title {
    text-align:center;
    font-size:36px;
    color:#77dd77;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌿 Digital Twin Millet System</div>", unsafe_allow_html=True)

# ===============================
# 🌾 DISEASE INFO (MATCHED)
# ===============================
disease_info = {
    "finger_smut": {
        "desc": "Fungal disease causing smut balls in finger millet.",
        "cause": "High humidity and infected seeds.",
        "treatment": "Apply Carbendazim fungicide and use certified seeds.",
        "fertilizer": "Apply balanced NPK fertilizer and avoid excess nitrogen."
    },
    "finger_wilt": {
        "desc": "Wilting due to fungal infection affecting roots.",
        "cause": "Soil-borne pathogens and poor drainage.",
        "treatment": "Use Trichoderma bio-control and fungicide.",
        "fertilizer": "Use organic compost and improve soil aeration."
    },
    "pearl_downy": {
        "desc": "Downy mildew disease in pearl millet.",
        "cause": "Cool humid climate and fungal spores.",
        "treatment": "Apply Metalaxyl fungicide.",
        "fertilizer": "Apply potassium-rich fertilizer."
    },
    "pearl_seedling": {
        "desc": "Seedling stage disease affecting growth.",
        "cause": "Poor soil and pathogens.",
        "treatment": "Seed treatment with fungicide.",
        "fertilizer": "Use phosphorus-rich fertilizer for root growth."
    }
}

# ===============================
# LOAD CLASS NAMES
# ===============================
class_names = np.load("class_names.npy", allow_pickle=True)

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
        out = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
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
# 🔥 HEATMAP
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
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = heatmap*0.4 + img_np
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Analysis", "📄 Report"])

# ===============================
# HOME
# ===============================
with tab1:
    st.write("🌾 Digital Twin system for millet disease detection and crop management.")

# ===============================
# ANALYSIS
# ===============================
with tab2:
    st.markdown("## 🔍 Analysis Dashboard")

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    col1, col2, col3 = st.columns(3)
    temp = col1.slider("Temperature", 10,50,25)
    humidity = col2.slider("Humidity",10,100,50)
    soil = col3.slider("Soil Moisture",10,100,50)

    if uploaded:
        img = Image.open(uploaded)

        left, right = st.columns([1,2])

        with left:
            st.image(img, use_column_width=True)
            if st.button("🔥 Show Heatmap"):
                st.image(generate_heatmap(img))

        pred = model.predict(preprocess(img))
        idx = np.argmax(pred)
        confidence = float(np.max(pred))

        disease = class_names[idx]
        display_name = disease.replace("_"," ").title()

        # Severity
        severity_index = (confidence*70)+(temp/50*10)+(humidity/100*10)+(soil/100*10)
        severity_index = min(severity_index,100)

        if severity_index > 70:
            level = "High"
        elif severity_index > 40:
            level = "Moderate"
        else:
            level = "Low"

        st.session_state.history.append((display_name, confidence))

        st.markdown("### 📊 Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Disease", display_name)
        m2.metric("Confidence", f"{confidence*100:.2f}%")
        m3.metric("Severity", level)

        st.progress(int(severity_index))

        # Charts
        st.plotly_chart(px.bar(x=class_names, y=pred[0]), use_container_width=True)
        top = np.argsort(pred[0])[-3:]
        st.plotly_chart(px.pie(values=pred[0][top], names=[class_names[i] for i in top]), use_container_width=True)

        # ===============================
        # 🧠 DIAGNOSIS
        # ===============================
        st.markdown("### 🧠 Diagnosis & Recommendation")

        if disease in disease_info:
            info = disease_info[disease]

            st.success(f"Disease: {display_name}")

            c1, c2 = st.columns(2)
            with c1:
                st.write(f"📌 Description: {info['desc']}")
                st.write(f"⚠ Cause: {info['cause']}")
            with c2:
                st.write(f"💊 Treatment: {info['treatment']}")
                st.write(f"🌾 Fertilizer: {info['fertilizer']}")

        else:
            st.warning("No data available")

# ===============================
# REPORT
# ===============================
with tab3:
    st.markdown("## 📄 Smart Report")

    if st.session_state.history:
        last = st.session_state.history[-1]

        st.metric("Last Disease", last[0])
        st.metric("Confidence", f"{last[1]*100:.2f}%")

        st.markdown("### 📊 History")
        for h in st.session_state.history:
            st.write(h)

    else:
        st.info("No predictions yet. Go to Analysis tab.")

# ===============================
# SIDEBAR
# ===============================
if st.session_state.role == "admin":
    st.sidebar.success("Admin Mode")
else:
    st.sidebar.info("User Mode")
