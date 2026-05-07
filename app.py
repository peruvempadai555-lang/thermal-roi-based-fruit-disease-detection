import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Layer, Conv2D, concatenate
import cv2
import requests

# ---------- DESKTOP CONFIGURATION ----------
st.set_page_config(
    page_title="Thermal Fruit Defect Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { padding: 1rem 2rem; }
    .stImage > img { border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    h1 { font-size: 2.8rem !important; margin-bottom: 1rem; }
    .stSuccess, .stMetric { font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)

# ---------- MODEL ACCURACY (from your validation) ----------
MODEL_ACCURACY = 0.92   # <-- change this if your accuracy is different

# ---------- CUSTOM LAYER ----------
class fire_module(Layer):
    def __init__(self, squeeze, expand, **kwargs):
        super().__init__(**kwargs)
        self.squeeze = squeeze
        self.expand = expand
        self.conv_squeeze = Conv2D(squeeze, (1,1), activation='tanh', padding='same')
        self.conv_expand1 = Conv2D(expand, (1,1), activation='tanh', padding='same')
        self.conv_expand3 = Conv2D(expand, (3,3), activation='tanh', padding='same')
    def call(self, inputs):
        s = self.conv_squeeze(inputs)
        e1 = self.conv_expand1(s)
        e3 = self.conv_expand3(s)
        return concatenate([e1, e3])
    def get_config(self):
        config = super().get_config()
        config.update({'squeeze': self.squeeze, 'expand': self.expand})
        return config

# ---------- PREPROCESSING FUNCTIONS ----------
def reduce_noise(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def extract_roi(image, roi_size=160):
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    half = roi_size // 2
    roi = image[cy-half:cy+half, cx-half:cx+half]
    if roi.size == 0:
        return image
    return reduce_noise(roi)

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img).astype(np.uint8)

    roi = extract_roi(img_np, roi_size=160)
    roi_resized = cv2.resize(roi, (224, 224))

    x = preprocess_input(roi_resized.astype(np.float32))
    x_min, x_max = x.min(), x.max()
    if x_max - x_min > 1e-8:
        x_norm = (x - x_min) / (x_max - x_min)
    else:
        x_norm = x - x_min

    x_uint8 = (x_norm * 255).astype(np.uint8)
    x_final = preprocess_input(x_uint8.astype(np.float32))

    return np.expand_dims(x_final, axis=0), Image.fromarray(roi_resized)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    url = "https://huggingface.co/spaces/M-Parames01/thermal-defect-model/resolve/main/fusion_resume_model_1.keras?download=true"
    response = requests.get(url)
    with open("fusion_resume_model_1.keras", "wb") as f:
        f.write(response.content)
    return tf.keras.models.load_model(
        "fusion_resume_model_1.keras",
        custom_objects={'fire_module': fire_module},
        compile=False
    )

model = load_model()
CLASS_NAMES = ['Major_Defect', 'Minor_Defect', 'No_Defect']

# ---------- UI ----------
st.title("🔥 Thermal ROI-Based Fruit Disease Detection")
st.markdown(f"**Model accuracy on validation set:** {MODEL_ACCURACY:.0%}")

uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        original = Image.open(uploaded_file).convert('RGB')
        st.image(original, caption="📸 Original Uploaded Image", use_container_width=True)
    with col2:
        roi_batch, roi_display = preprocess_image(uploaded_file)
        st.image(roi_display, caption="🔍 Extracted ROI (after preprocessing)", use_container_width=True)

    st.markdown("---")
    with st.spinner("Analyzing thermal pattern..."):
        pred = model.predict(roi_batch)
        pred_class = CLASS_NAMES[np.argmax(pred[0])]
        raw_confidence = np.max(pred[0])

        # Scale by model accuracy
        displayed_confidence = raw_confidence * MODEL_ACCURACY

    col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
    with col_res2:
        st.success(f"### 🧠 Prediction: **{pred_class}**")
        st.metric("Calibrated Confidence", f"{displayed_confidence:.2%}",
                  help=f"Raw confidence was {raw_confidence:.2%} × model accuracy {MODEL_ACCURACY:.0%}")
        if pred_class == "No_Defect":
            st.balloons()
        elif pred_class == "Minor_Defect":
            st.info("⚠️ Minor defect detected – monitor fruit condition.")
        else:
            st.error("❌ Major defect detected – discard fruit.")
else:
    st.info("👈 Please upload a thermal fruit image (JPG, PNG, JPEG) to start detection.")
