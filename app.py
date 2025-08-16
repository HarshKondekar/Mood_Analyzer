# Mood Detection - FER2013 (PyTorch + MongoDB / PyMongo)
# -------------------------------------------------------
# Firebase removed. Replaced with MongoDB for metadata + GridFS for image storage.

# =============================
# Streamlit Config
# =============================
import streamlit as st
st.set_page_config(page_title="Mood Detection - FER2013", layout="centered")

"""
FER2013 Mood Detection App with PyTorch + MongoDB
=================================================
- Upload image OR use live camera
- Predict emotion using PyTorch model
- If incorrect, user can label correct emotion → Stored in MongoDB (image in GridFS)
"""

# =============================
# Imports
# =============================
from dotenv import load_dotenv
import os
import io
import base64
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from pymongo import MongoClient
import gridfs
from bson import ObjectId
import certifi  # TLS/SSL for Atlas

# Optional: OpenCV for camera processing
try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None

# =============================
# Load environment variables
# =============================
load_dotenv()  # automatically loads .env from project root

# =============================
# Mongo Config
# =============================
MONGO_URI = os.getenv("MONGO_URI") or "mongodb+srv://harshkondekar:vEmSS5mpmGWvBSpo@moodanalyzer.2bnqwd0.mongodb.net/mood_detection?retryWrites=true&w=majority"
MONGO_URI_FALLBACK = "mongodb://localhost:27017/"
MONGO_DB_NAME = "mood_detection"
FEEDBACK_COLLECTION = "feedback"
GRIDFS_BUCKET_NAME = "feedback_images"

# =============================
# Device & Classes
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
st.caption(f"Using device: {DEVICE}")

# =============================
# Model Definition
# =============================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AdvancedMoodClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        from torchvision import models
        backbone = models.efficientnet_b4(pretrained=True)
        self.features = backbone.features
        self.attention = SEBlock(1792)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1792, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Dropout(0.4), nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Dropout(0.2), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.classifier(x)

# =============================
# Load Model
# =============================
CKPT_PATH = "final_mood_classifier_HK.pth"

@st.cache_resource
def load_model():
    try:
        model = AdvancedMoodClassifier(len(CLASS_NAMES)).to(DEVICE)
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

MODEL = load_model()

# =============================
# Transforms & Prediction
# =============================
IMG_SIZE = 224
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image: Image.Image) -> Tuple[Optional[int], Optional[np.ndarray]]:
    if MODEL is None:
        return None, None
    img_tensor = VAL_TRANSFORMS(image.convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(img_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
    return pred_idx, probs

def plot_probs(probs: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(CLASS_NAMES, probs)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    plt.tight_layout()
    st.pyplot(fig)

# =============================
# Mongo Init
# =============================
@st.cache_resource(show_spinner=False)
def init_mongo():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000, tlsCAFile=certifi.where())
        client.admin.command("ping")
        db = client[MONGO_DB_NAME]
        fs = gridfs.GridFS(db, collection=GRIDFS_BUCKET_NAME)
        st.success("✅ MongoDB connected successfully (Atlas)")
        return client, db, fs
    except Exception:
        try:
            client = MongoClient(MONGO_URI_FALLBACK, serverSelectionTimeoutMS=5000)
            db = client[MONGO_DB_NAME]
            fs = gridfs.GridFS(db, collection=GRIDFS_BUCKET_NAME)
            st.warning("Connected to local MongoDB instead of Atlas")
            return client, db, fs
        except Exception as e:
            st.error(f"❌ MongoDB connection failed: {e}")
            return None, None, None

CLIENT, DB, FS = init_mongo()
FEEDBACK_COL = DB[FEEDBACK_COLLECTION] if DB else None

# =============================
# Feedback Functions
# =============================
def save_feedback(image: Image.Image, predicted: str, correct: str, probs: Optional[np.ndarray] = None) -> bool:
    if FS is None or FEEDBACK_COL is None:
        st.error("❌ MongoDB not initialized. Cannot save feedback.")
        return False
    try:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"feedback_{ts}.jpg"
        img_buffer = io.BytesIO()
        image.convert('RGB').save(img_buffer, format='JPEG', quality=95)
        gridfs_id = FS.put(img_buffer.getvalue(), filename=filename, contentType='image/jpeg')
        feedback_data = {
            'timestamp_utc': datetime.utcnow().isoformat(),
            'predicted_emotion': predicted,
            'correct_emotion': correct,
            'gridfs_id': gridfs_id,
            'image_filename': filename,
            'user_id': st.session_state.get('user_id', 'anonymous')
        }
        if probs is not None:
            feedback_data['probs'] = [float(p) for p in probs]
        result = FEEDBACK_COL.insert_one(feedback_data)
        st.success(f"✅ Feedback saved! Doc ID: {result.inserted_id}")
        return True
    except Exception as e:
        st.error(f"❌ Error saving feedback: {e}")
        return False

def load_feedback_image(gridfs_id: ObjectId) -> Optional[Image.Image]:
    if FS is None:
        return None
    try:
        data = FS.get(gridfs_id).read()
        return Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as e:
        st.warning(f"Failed to load image from GridFS: {e}")
        return None

def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode()

# =============================
# Streamlit UI
# =============================
st.title("😃 FER2013 Mood Detection")

if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if CLIENT and DB and FS:
    st.success("✅ MongoDB connected")
else:
    st.error("❌ MongoDB not connected")

st.subheader("Choose Input Method")
col1, col2 = st.columns(2)
with col1:
    if st.button("📤 Upload Image", use_container_width=True):
        st.session_state.mode = 'upload'
        st.session_state.feedback_given = False
with col2:
    if st.button("📸 Live Camera", use_container_width=True):
        st.session_state.mode = 'camera'
        st.session_state.feedback_given = False

img = None
if st.session_state.mode == 'upload':
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"], key="upload_img")
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)
elif st.session_state.mode == 'camera':
    st.subheader("Camera Input")
    camera_image = st.camera_input("Take a photo")
        if camera_image:
        img = Image.open(camera_image).convert('RGB')
        st.image(img, caption="Camera Image", use_container_width=True)

if img is not None:
    if st.button("🔮 Predict Mood", use_container_width=True):
        if MODEL is None:
            st.error("Model not loaded. Please check the model file.")
        else:
            with st.spinner("Analyzing emotion..."):
                pred_idx, probs = predict(img)
                if pred_idx is not None:
                    st.session_state.pred_idx = pred_idx
                    st.session_state.probs = probs.tolist()
                    st.session_state.image = img
                    st.session_state.feedback_given = False
                    st.success("Prediction complete!")
                else:
                    st.error("Prediction failed.")

if 'pred_idx' in st.session_state and 'probs' in st.session_state:
    pred_idx = st.session_state.pred_idx
    probs = np.array(st.session_state.probs)
    predicted_emotion = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]

    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Emotion", predicted_emotion)
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")

    st.subheader("Probability Distribution")
    plot_probs(probs)

    if not st.session_state.feedback_given:
        st.subheader("Feedback")
        st.write("Was this prediction correct?")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ Yes, it's correct!", use_container_width=True):
                with st.spinner("Saving feedback..."):
                    success = save_feedback(
                        st.session_state.image,
                        predicted_emotion,
                        predicted_emotion,
                        probs=probs
                    )
                if success:
                    st.success("Thanks for confirming!")
                    st.session_state.feedback_given = True
                    st.rerun()
        with col_no:
            if st.button("❌ No, it's wrong", use_container_width=True):
                st.session_state.show_correction = True
                st.rerun()

        if st.session_state.get('show_correction', False):
            st.write("Please select the correct emotion:")
            correct_emotion = st.selectbox(
                "Correct emotion:",
                CLASS_NAMES,
                key="correct_emotion_select"
            )
            if st.button("💾 Save Correction", use_container_width=True):
                with st.spinner("Saving feedback..."):
                    success = save_feedback(
                        st.session_state.image,
                        predicted_emotion,
                        correct_emotion,
                        probs=probs
                    )
                if success:
                    st.success("✅ Thank you! Your feedback was saved.")
                    st.session_state.feedback_given = True
                    st.session_state.show_correction = False
                    st.rerun()
                else:
                    st.error("❌ Failed to save feedback.")

# =============================
# Sidebar Info
# =============================
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a deep learning model trained on the FER2013 dataset to detect emotions."
)
st.sidebar.title("How to use")
st.sidebar.markdown(
    """
    1. Upload or capture an image  
    2. Click 'Predict Mood'  
    3. Provide feedback  
    4. Feedback stored in MongoDB (images in GridFS)  
    """
)
