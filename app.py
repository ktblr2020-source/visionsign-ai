import cv2
import streamlit as st
import numpy as np
import os
import urllib.request
import time
from collections import deque
from sklearn.neural_network import MLPClassifier

# Page Configuration
st.set_page_config(
    page_title="VisionSign AI ✨",
    page_icon="🤖",
    layout="wide"
)

# Custom Fun & Clean Styling
st.markdown("""
    <style>
    .stButton>button {
        background: linear-gradient(135deg, #f43f5e 0%, #fbbf24 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.7rem 1.4rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 20px rgba(244, 63, 94, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.02) translateY(-2px);
        box-shadow: 0 6px 25px rgba(244, 63, 94, 0.6);
    }
    .fun-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid rgba(244, 63, 94, 0.3);
        padding: 24px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        color: #f8fafc;
    }
    .neon-text {
        background: linear-gradient(45deg, #f43f5e, #fbbf24, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
    }
    </style>
""", unsafe_allow_html=True)

# App Header & Branding Credit
st.markdown("<h1 style='text-align: center;'>🤖 <span class='neon-text'>VisionSign Playground</span> ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem;'>High-Accuracy Neural Sign Recognition Engine</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: right; color: #64748b; font-size: 0.85rem; margin-top: -15px;'><i>Engineered by Kevin and team</i></p>", unsafe_allow_html=True)
st.markdown("---")

# Load and Train MLP Neural Network Classifier on Normalized Data
@st.cache_resource
def load_and_train_model():
    data_dir = './dataset'
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        return None, "No dataset found! Record normalized data using collect.py first."
    
    X, y = [], []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.endswith('.npy'):
                    file_path = os.path.join(label_path, file)
                    landmarks = np.load(file_path)
                    X.append(landmarks)
                    y.append(label)
                    
    if len(X) == 0:
        return None, "Dataset folders are empty!"
        
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    model.fit(np.array(X), np.array(y))
    return model, "Success"

model, status_msg = load_and_train_model()

# Session State for Translated Text Output
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""

# Layout Columns
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown("### 📷 Live Camera Stage")
    run_camera = st.checkbox("🚀 Launch High-Accuracy AI", value=False)
    camera_placeholder = st.empty()
    status_placeholder = st.empty()

with col_right:
    st.markdown("### 💬 Live Typed Message")
    
    text_box = st.empty()
    text_box.markdown(f'<div class="fun-card" style="min-height: 130px; font-size: 1.8rem; color: #38bdf8; word-break: break-all; font-family: monospace;">{st.session_state.translated_text or "👋 Waiting for your awesome signs..."}</div>', unsafe_allow_html=True)
    
    st.write("") 
    # Control Buttons
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        clear_btn = st.button("🧹 Clear")
    with b_col2:
        space_btn = st.button("␣ Space")
    with b_col3:
        backspace_btn = st.button("⬅️ Back")

if clear_btn:
    st.session_state.translated_text = ""
    st.rerun()
if space_btn:
    st.session_state.translated_text += " "
    st.rerun()
if backspace_btn:
    st.session_state.translated_text = st.session_state.translated_text[:-1]
    st.rerun()

# Ensure Model is Ready
if model is None:
    st.warning(f"⚠️ {status_msg}")
else:
    MODEL_PATH = "hand_landmarker.task"
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import mediapipe as mp

    if run_camera:
        cap = cv2.VideoCapture(0)
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )
        
        start_time = time.time()
        prev_wrist_x = 0
        
        # Letter Buffer & Smoothing Variables
        last_char = ""
        hold_start_time = None
        HOLD_THRESHOLD = 1.0  # Seconds to hold sign before auto-typing
        prediction_buffer = deque(maxlen=7)  # Smoothing window
        
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened() and run_camera:
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((time.time() - start_time) * 1000)
                
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                predicted_char = "🔍 Searching for hand..."
                
                if detection_result.hand_landmarks:
                    hand_landmarks = detection_result.hand_landmarks[0]
                    
                    # Relative Coordinate Normalization (Wrist as Origin)
                    temp_coords = []
                    base_x, base_y, base_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
                    for lm in hand_landmarks:
                        temp_coords.append([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                    
                    max_dist = max([np.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2) for pt in temp_coords])
                    if max_dist == 0: 
                        max_dist = 1.0

                    landmarks = []
                    for pt in temp_coords:
                        landmarks.extend([pt[0]/max_dist, pt[1]/max_dist, pt[2]/max_dist])
                    
                    # Motion tracking delta
                    wrist_x = hand_landmarks[0].x
                    motion_delta = abs(wrist_x - prev_wrist_x)
                    prev_wrist_x = wrist_x

                    # Model prediction
                    raw_pred = model.predict(np.array([landmarks]))[0]
                    
                    # Special Motion Overrides for 'J' and 'Z'
                    if raw_pred == 'I' and motion_delta > 0.04:
                        raw_pred = 'J'
                    elif raw_pred == 'D' and motion_delta > 0.03:
                        raw_pred = 'Z'
                    
                    # Majority vote smoothing buffer
                    prediction_buffer.append(raw_pred)
                    predicted_char = max(set(prediction_buffer), key=prediction_buffer.count)

                    # Auto-typing logic
                    if predicted_char == last_char:
                        if hold_start_time is None:
                            hold_start_time = time.time()
                        elif (time.time() - hold_start_time) >= HOLD_THRESHOLD:
                            st.session_state.translated_text += predicted_char
                            text_box.markdown(f'<div class="fun-card" style="min-height: 130px; font-size: 1.8rem; color: #38bdf8; word-break: break-all; font-family: monospace;">{st.session_state.translated_text}</div>', unsafe_allow_html=True)
                            hold_start_time = None
                    else:
                        last_char = predicted_char
                        hold_start_time = time.time()

                    # Draw landmark nodes on hand
                    for lm in hand_landmarks:
                        h, w, _ = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (251, 191, 36), -1)
                else:
                    last_char = ""
                    hold_start_time = None
                    prediction_buffer.clear()

                # Stream live camera feed
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Feedback Banner
                status_placeholder.markdown(f"<h2 style='text-align: center; color: #fbbf24;'>🎯 Detected Sign: <span style='color: #f43f5e;'>{predicted_char}</span></h2>", unsafe_allow_html=True)
                
        cap.release()