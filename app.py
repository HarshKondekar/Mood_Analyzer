# Mood Detection - FER2013 (PyTorch + MongoDB / PyMongo)
# -------------------------------------------------------
# Firebase removed. Replaced with MongoDB for metadata + GridFS for image storage.
# Instructions:
#   1. Set your Mongo connection string in one of the following ways (checked in this order):
#        - st.secrets["MONGO_URI"]  (preferred for Streamlit Cloud / sharing)
#        - Environment variable MONGO_URI
#        - Hard-coded fallback in MONGO_URI_FALLBACK below (NOT recommended for production)
#   2. Ensure the target database name exists or can be created (MONGO_DB_NAME).
#   3. Ensure you have network access / IP allowlist for the MongoDB deployment.
#
# Data model:
#   - Images are stored in GridFS bucket named GRIDFS_BUCKET_NAME (default: "feedback_images").
#   - Feedback metadata stored in collection FEEDBACK_COLLECTION (default: "feedback"). Each doc:
#       {
#         _id: ObjectId,
#         timestamp_utc: ISO8601 str,
#         predicted_emotion: str,
#         correct_emotion: str,
#         gridfs_id: ObjectId,            # pointer to the image file in GridFS
#         image_filename: str,            # original file-ish name we generated
#         user_id: str,
#         probs: [float,...] (optional)   # full probability vector if available
#       }
#
# Retrieval demo utilities (optional) are included at bottom (collapsed behind a debug checkbox).
#
# -------------------------------------------------------

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
# Retrieve secrets
mongo_uri = st.secrets["MONGO"]["URI"]
db_name = st.secrets["MONGO"]["DB_NAME"]
import os
import io
import json
import base64
from datetime import datetime
from typing import Optional, Dict, Tuple

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

# Optional: OpenCV for camera processing
try:
    import cv2  # noqa: F401
except ImportError:  # not fatal
    cv2 = None

# =============================
# Mongo Config
# =============================
# Order of precedence for connection string:
# 1. st.secrets["MONGO_URI"]
# 2. os.environ["MONGO_URI"]
# 3. fallback literal below
MONGO_URI_FALLBACK = "mongodb://localhost:27017/"  # dev default; change/remove in prod
MONGO_DB_NAME = st.secrets["MONGO"]["DB_NAME"]
FEEDBACK_COLLECTION = "feedback"
GRIDFS_BUCKET_NAME = "feedback_images"  # GridFS bucket prefix (creates <bucket>.files & <bucket>.chunks)


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
CKPT_PATH = "final_mood_classifier HK.pth"  # update path as needed

@st.cache_resource
def load_model():
    try:
        model = AdvancedMoodClassifier(len(CLASS_NAMES)).to(DEVICE)
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:  # surface error to user
        st.error(f"Error loading model: {e}")
        return None

MODEL = load_model()

# =============================
# Transforms
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
    """Initialize a cached Mongo connection + GridFS handle.

    Returns
    -------
    (client, db, fs) or (None, None, None) on failure.
    """

    uri = st.secrets["MONGO"]["URI"]
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Trigger a server selection to fail fast if unreachable
        client.admin.command('ping')
        db = client[MONGO_DB_NAME]
        fs = gridfs.GridFS(db, collection=GRIDFS_BUCKET_NAME)
        st.success("MongoDB connected successfully!")
        return client, db, fs
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        return None, None, None


CLIENT, DB, FS = init_mongo()

# Convenience collection handle (may be None)
if DB is not None:
    FEEDBACK_COL = DB[FEEDBACK_COLLECTION]
else:
    FEEDBACK_COL = None


# =============================
# Save Feedback (Mongo)
# =============================

def save_feedback(image: Image.Image, predicted: str, correct: str, probs: Optional[np.ndarray] = None) -> bool:
    """Save user feedback to MongoDB.

    Parameters
    ----------
    image : PIL.Image
        User image to persist.
    predicted : str
        Model-predicted emotion.
    correct : str
        User-corrected emotion.
    probs : np.ndarray, optional
        Probability vector from model.

    Returns
    -------
    bool
        True if saved successfully, else False.
    """
    if FS is None or FEEDBACK_COL is None:
        st.error("MongoDB not initialized. Cannot save feedback.")
        return False

    try:
        # Unique timestamp-based name
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"feedback_{ts}.jpg"

        # Convert image to bytes
        img_buffer = io.BytesIO()
        rgb_image = image.convert('RGB')
        rgb_image.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()

        # Store in GridFS
        gridfs_id = FS.put(img_bytes, filename=filename, contentType='image/jpeg')

        # Build metadata doc
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
        print("RESULT DEKHO BC" , result)
        st.success(f"Feedback saved successfully! Doc ID: {result.inserted_id}")
        return True

    except Exception as e:
        st.error(f"Error saving feedback: {e}")
        return False


# =============================
# Helper: load image from GridFS (debug / admin)
# =============================

def load_feedback_image(gridfs_id: ObjectId) -> Optional[Image.Image]:
    if FS is None:
        return None
    try:
        data = FS.get(gridfs_id).read()
        return Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as e:  # debug only
        st.warning(f"Failed to load image from GridFS: {e}")
        return None


# =============================
# Helper function to convert image to base64 (UI convenience)
# =============================

def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode()


# =============================
# Streamlit UI
# =============================
st.title("😃 FER2013 Mood Detection")

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Display Mongo status
if CLIENT is not None and DB is not None and FS is not None:
    st.success("✅ MongoDB connected successfully")
else:
    st.error("❌ MongoDB not connected")

# Mode Selection
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

# Image Input
img = None
if st.session_state.mode == 'upload':
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

elif st.session_state.mode == 'camera':
    st.subheader("Camera Input")
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        img = Image.open(camera_image).convert('RGB')
        st.image(img, caption="Camera Image", use_container_width=True)

# Prediction trigger
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

# Display Results
if 'pred_idx' in st.session_state and 'probs' in st.session_state:
    pred_idx = st.session_state.pred_idx
    probs = np.array(st.session_state.probs)
    predicted_emotion = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]

    st.subheader("Prediction Results")

    # Display prediction with confidence
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Emotion", predicted_emotion)
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")

    # Plot probabilities
    st.subheader("Probability Distribution")
    plot_probs(probs)

    # Feedback Section
    if not st.session_state.feedback_given:
        st.subheader("Feedback")
        st.write("Was this prediction correct?")

        col_yes, col_no = st.columns(2)

        with col_yes:
            if st.button("✅ Yes, it's correct!", use_container_width=True):
                # Save auto-confirmed feedback where correct == predicted
                with st.spinner("Saving feedback..."):
                    success = save_feedback(
                        st.session_state.image,
                        predicted_emotion,
                        predicted_emotion,  # confirmed
                        probs=probs,
                    )
                if success:
                    st.success("Thank you for confirming! This helps improve our model.")
                    st.session_state.feedback_given = True
                    st.rerun()

        with col_no:
            if st.button("❌ No, it's wrong", use_container_width=True):
                st.session_state.show_correction = True
                st.rerun()

        # Show correction interface
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
                        probs=probs,
                    )

                if success:
                    st.success("✅ Thank you! Your feedback has been saved and will help improve our model.")
                    st.session_state.feedback_given = True
                    st.session_state.show_correction = False
                    st.rerun()
                else:
                    st.error("❌ Failed to save feedback. Please try again.")

# =============================
# Sidebar Info
# =============================
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a deep learning model trained on the FER2013 dataset to detect emotions in facial expressions. "
    "The model can recognize 7 different emotions: angry, disgust, fear, happy, neutral, sad, and surprise."
)

st.sidebar.title("How to use")
st.sidebar.markdown(
    """
    1. Choose to upload an image or use your camera  
    2. Click 'Predict Mood' to analyze the emotion  
    3. Provide feedback to help improve the model  
    4. Your feedback is stored in MongoDB (image in GridFS)  
    """
)

# Debug information (optional)
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.subheader("Debug Information")
    st.sidebar.write(f"Device: {DEVICE}")
    st.sidebar.write(f"Model loaded: {MODEL is not None}")
    st.sidebar.write(f"Mongo connected: {CLIENT is not None and DB is not None and FS is not None}")
    st.sidebar.write(f"Session state keys: {list(st.session_state.keys())}")

    if FEEDBACK_COL is not None:
        st.sidebar.write("Recent feedback docs:")
        try:
            recent = list(FEEDBACK_COL.find().sort('timestamp_utc', -1).limit(5))
            for doc in recent:
                st.sidebar.json({k: str(v) if isinstance(v, ObjectId) else v for k, v in doc.items()})
        except Exception as e:
            st.sidebar.write(f"(error fetching feedback) {e}")

        # Optionally display the latest stored image
        if st.sidebar.button("Show latest stored image"):
            try:
                latest = FEEDBACK_COL.find().sort('timestamp_utc', -1).limit(1)
                latest_list = list(latest)
                if latest_list:
                    doc = latest_list[0]
                    grid_id = doc.get('gridfs_id')
                    if grid_id:
                        pil_img = load_feedback_image(grid_id)
                        if pil_img is not None:
                            st.sidebar.image(pil_img, caption=f"Latest image ({doc.get('predicted_emotion')})")
                        else:
                            st.sidebar.write("Could not load image from GridFS.")
                else:
                    st.sidebar.write("No feedback docs yet.")
            except Exception as e:
                st.sidebar.write(f"(error loading latest image) {e}")