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
body {background-color:#0f172a;}
.main-title {
    text-align:center;
    font-size:40px;
    color:#77dd77;
    font-weight:bold;
    margin-bottom:20px;
}
.card {
    background:#111827;
    padding:20px;
    border-radius:15px;
    border:1px solid #2d3748;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌿 Digital Twin Millet System</div>", unsafe_allow_html=True)

# ===============================
# 🌾 DISEASE INFO
# ===============================
disease_info = {
    "finger_smut": {
        "desc": "Fungal disease forming smut balls that reduce grain quality.",
        "cause": "High humidity and infected seeds.",
        "treatment": "Apply Carbendazim and use certified seeds.",
        "fertilizer": "Balanced NPK improves plant resistance."
    },
    "finger_wilt": {
        "desc": "Wilting due to soil-borne fungal infection affecting roots.",
        "cause": "Poor drainage and infected soil.",
        "treatment": "Apply Trichoderma bio-control.",
        "fertilizer": "Organic compost improves soil health."
    },
    "pearl_downy": {
        "desc": "Downy mildew affecting leaf growth and productivity.",
        "cause": "Cool humid environment.",
        "treatment": "Use Metalaxyl fungicide.",
        "fertilizer": "Potassium boosts plant immunity."
    },
    "pearl_seedling": {
        "desc": "Early-stage disease affecting seedling development.",
        "cause": "Soil pathogens and poor seed quality.",
        "treatment": "Seed treatment with fungicide.",
        "fertilizer": "Phosphorus enhances root development."
    }
}

# ===============================
# LOAD CLASS NAMES
# ===============================
class_names = np.load("class_names.npy", allow_pickle=True)

# ===============================
# MODEL LOADING (STABLE)
# ===============================
@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model
    try:
        return load_model("fixed_model.h5", compile=False)
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
# HEATMAP
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

    heatmap = heatmap.numpy()
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap)+1e-8)
    heatmap = cv2.resize(heatmap, (224,224))

    img_np = np.array(img.resize((224,224)))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = heatmap*0.4 + img_np
    return np.clip(overlay,0,255).astype(np.uint8)

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["🏠 Home","📊 Analysis","📄 Report"])

# ===============================
# HOME
# ===============================
with tab1:
    st.write("🌾 Smart system for millet disease detection and crop management.")

# ===============================
# ANALYSIS
# ===============================
with tab2:
    st.markdown("## 🔍 Analysis Dashboard")

    uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    col1, col2, col3 = st.columns(3)
    temp = col1.slider("🌡 Temperature",10,50,25)
    humidity = col2.slider("💧 Humidity",10,100,50)
    soil = col3.slider("🌱 Soil Moisture",10,100,50)

    if uploaded:
        img = Image.open(uploaded)

        left, right = st.columns([1,2])

        with left:
            st.image(img, use_container_width=True)
            if st.button("🔥 Show Heatmap"):
                st.image(generate_heatmap(img))
                st.info("Model highlights infected regions.")

        pred = model.predict(preprocess(img))
        idx = np.argmax(pred)
        confidence = float(np.max(pred))

        disease = class_names[idx]
        display = disease.replace("_"," ").title()

        st.session_state.history.append((display, confidence))

        with right:
            st.markdown(f"""
            <div class="card">
                <h2 style="color:#77dd77;">🌱 {display}</h2>
                <h3 style="color:#4fc3f7;">📊 {confidence*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            # Charts
            st.markdown("### 📊 Model Insights")

            fig1 = px.bar(
                x=[c.replace("_"," ").title() for c in class_names],
                y=pred[0],
                title="Prediction Confidence"
            )
            st.plotly_chart(fig1, use_container_width=True)

            top = np.argsort(pred[0])[-3:]
            fig2 = px.pie(
                values=pred[0][top],
                names=[class_names[i].replace("_"," ").title() for i in top],
                title="Top Predictions"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Explanation
            st.markdown("### 🧠 Diagnosis & Recommendation")

            if disease in disease_info:
                info = disease_info[disease]

                st.markdown(f"""
                <div class="card">
                <p><b>📌 Description:</b> {info['desc']}</p>
                <p><b>⚠ Cause:</b> {info['cause']}</p>
                <p><b>💊 Treatment:</b> {info['treatment']}</p>
                <p><b>🌾 Fertilizer:</b> {info['fertilizer']}</p>
                </div>
                """, unsafe_allow_html=True)

# ===============================
# REPORT
# ===============================
with tab3:
    st.markdown("## 📄 Smart Report")

    if st.session_state.history:
        d,c = st.session_state.history[-1]

        st.metric("🌱 Disease", d)
        st.metric("📊 Confidence", f"{c*100:.2f}%")

        diseases = [h[0] for h in st.session_state.history]
        conf = [h[1]*100 for h in st.session_state.history]

        st.plotly_chart(px.line(x=list(range(len(diseases))), y=conf, markers=True),
                        use_container_width=True)

        st.markdown("### 📋 History")
        for i,(d,c) in enumerate(st.session_state.history):
            st.write(f"{i+1}. {d} — {c*100:.2f}%")

    else:
        st.info("No predictions yet.")
