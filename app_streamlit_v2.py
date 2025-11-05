# -------------------------------
# AI Posture Detector (Simplified)
# -------------------------------

import av
import cv2
import math
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
from datetime import datetime
import queue
import threading
import time
import pyttsx3
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------
# Streamlit Config
# ---------------------------------
st.set_page_config(layout="wide", page_title="AI Posture Dashboard")

# TTS engine (for voice feedback)
tts_engine = pyttsx3.init()
def speak_async(text: str):
    threading.Thread(target=lambda: (tts_engine.say(text), tts_engine.runAndWait()), daemon=True).start()

# Helper: calculate angle between 3 points
def angle_between_points(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return math.degrees(math.acos(cosang))

def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

# ---------------------------------
# Sidebar UI
# ---------------------------------
st.sidebar.title("Session Controls")
session_name = st.sidebar.text_input("Session Name:", value=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
start_button = st.sidebar.button("Start/Restart Session")
download_button = st.sidebar.button("Download CSV")
save_logs_button = st.sidebar.button("Save Log")

data_q = queue.Queue()

# ---------------------------------
# Mediapipe Pose Setup
# ---------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------------------
# Session State
# ---------------------------------
if "session_df" not in st.session_state:
    st.session_state.session_df = pd.DataFrame(columns=["timestamp", "posture", "spine_angle", "neck_angle"])

# ---------------------------------
# Layout Columns
# ---------------------------------
left_col, mid_col, right_col = st.columns([1.4, 0.9, 1.2])

with left_col:
    st.header("Live Camera")

with mid_col:
    st.header("Metrics")
    posture_metric = st.empty()
    spine_metric = st.empty()
    neck_metric = st.empty()
    st.markdown("---")
    st.subheader("Advice")
    advice_box = st.empty()

with right_col:
    st.header("Charts")
    posture_chart = st.empty()
    st.markdown("**Recent Log**")
    log_table = st.empty()

# ---------------------------------
# Posture Decision
# ---------------------------------
def posture_from_angles(spine_angle, neck_angle):
    if spine_angle > 165 and neck_angle > 160:
        return "Excellent"
    if spine_angle > 150 and neck_angle > 145:
        return "Good"
    if spine_angle > 135:
        return "Leaning"
    return "Poor"

# ---------------------------------
# Video Transformer
# ---------------------------------
class Transformer(VideoTransformerBase):
    def __init__(self):
        self.angle_buffer = []
        self.last_spoken = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose_model.process(img_rgb)

        spine_angle = 0.0
        neck_angle = 0.0
        posture_label = "No person"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def to_xy(l): return (int(l.x * w), int(l.y * h))

            ls, rs = to_xy(l[mp_pose.PoseLandmark.LEFT_SHOULDER.value]), to_xy(l[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            lh, rh = to_xy(l[mp_pose.PoseLandmark.LEFT_HIP.value]), to_xy(l[mp_pose.PoseLandmark.RIGHT_HIP.value])
            lk, rk = to_xy(l[mp_pose.PoseLandmark.LEFT_KNEE.value]), to_xy(l[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            nose = to_xy(l[mp_pose.PoseLandmark.NOSE.value])

            shoulders_mid = midpoint(ls, rs)
            hips_mid = midpoint(lh, rh)
            knees_mid = midpoint(lk, rk)

            try:
                spine_angle = angle_between_points(shoulders_mid, hips_mid, knees_mid)
                neck_angle = angle_between_points(nose, shoulders_mid, hips_mid)
            except Exception:
                spine_angle = 0.0
                neck_angle = 0.0

            self.angle_buffer.append(spine_angle)
            if len(self.angle_buffer) > 8:
                self.angle_buffer.pop(0)
            smoothed_spine = float(np.mean(self.angle_buffer))

            posture_label = posture_from_angles(smoothed_spine, neck_angle)

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(img, f"Spine: {int(smoothed_spine)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(img, f"Neck: {int(neck_angle)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)
            cv2.putText(img, f"Posture: {posture_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,180,0), 2)

        # Save data
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "posture": posture_label,
            "spine_angle": float(spine_angle),
            "neck_angle": float(neck_angle)
        }
        if data_q.qsize() < 5000:
            data_q.put_nowait(row)

        # Voice feedback
        now = time.time()
        if posture_label in ["Poor", "Leaning"] and now - self.last_spoken > 8.0:
            speak_async("Please sit upright for better posture.")
            self.last_spoken = now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------------
# Start Button
# ---------------------------------
if start_button:
    st.session_state.session_df = pd.DataFrame(columns=["timestamp", "posture", "spine_angle", "neck_angle"])
    while not data_q.empty():
        try: data_q.get_nowait()
        except: break

webrtc_ctx = webrtc_streamer(
    key="posture-only-demo",
    video_transformer_factory=Transformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

# ---------------------------------
# Update UI
# ---------------------------------
def update_ui_from_queue():
    updated = False
    while not data_q.empty():
        try:
            row = data_q.get_nowait()
        except queue.Empty:
            break
        st.session_state.session_df = pd.concat([st.session_state.session_df, pd.DataFrame([row])], ignore_index=True)
        updated = True

    if updated and len(st.session_state.session_df) > 0:
        last = st.session_state.session_df.iloc[-1]

        posture_metric.metric("Posture", last["posture"])
        spine_metric.metric("Spine Angle", int(last["spine_angle"]))
        neck_metric.metric("Neck Angle", int(last["neck_angle"]))

        advice = ""
        if last["posture"] in ["Poor", "Leaning"]:
            advice = "⚠️ Sit upright and keep shoulders relaxed."
        else:
            advice = "✅ Good posture maintained."

        advice_box.info(advice)

        df = st.session_state.session_df.copy().tail(200)
        posture_counts = df["posture"].value_counts()
        posture_chart.bar_chart(posture_counts)
        log_table.dataframe(df.tail(10))

if webrtc_ctx and webrtc_ctx.state.playing:
    update_ui_from_queue()

# ---------------------------------
# Download CSV
# ---------------------------------
if download_button:
    csv = st.session_state.session_df.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name=f"{session_name}.csv", mime="text/csv")

if save_logs_button:
    logs_dir = Path("session_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    filename = logs_dir / f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    st.session_state.session_df.to_csv(filename, index=False)
    st.success(f"Saved to {filename}")
