# app_streamlit.py
import av
import cv2
import math
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
from fer import FER
from datetime import datetime
import matplotlib.pyplot as plt
import queue
import threading

st.set_page_config(layout="wide", page_title="AI Posture & Emotion Dashboard")

# ---- Sidebar ----
st.sidebar.title("Session Controls")
st.sidebar.markdown("Start the webcam and collect a session log with posture + emotion.")
session_name = st.sidebar.text_input("Session name (optional):", value="session1")
start_button = st.sidebar.button("Start/Restart Session")
download_button = st.sidebar.button("Download CSV")

# Session data storage (thread-safe queue)
data_q = queue.Queue()

# Initialize detectors once
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# Initialize emotion detector (OpenCV backend, slightly faster)
emotion_detector = FER(mtcnn=False, scale_factor=0.9)


# WebRTC media stream settings (video on, audio off)
WEBRTC_CLIENT_SETTINGS = {
    "media_stream_constraints": {"video": True, "audio": False}
}


# Simple smoothing buffer for angles
ANGLE_BUFFER_LEN = 8

# Logging dataframe (in-memory)
if "session_df" not in st.session_state:
    st.session_state.session_df = pd.DataFrame(columns=["timestamp", "posture", "angle", "emotion", "confidence"])

# ---- Layout ----
left_col, mid_col, right_col = st.columns([1.2, 1, 1.3])

with left_col:
    st.header("Live Camera")
    # We will show the webrtc component here; transformer defined below
    st_webrtc = None

with mid_col:
    st.header("Metrics")
    posture_metric = st.empty()
    angle_metric = st.empty()
    emotion_metric = st.empty()
    confidence_metric = st.empty()
    st.markdown("---")
    st.subheader("Advice")
    advice_box = st.empty()

with right_col:
    st.header("Charts")
    posture_chart = st.empty()
    emotion_chart = st.empty()
    st.markdown("*Session log (last 10 rows)*")
    log_table = st.empty()

# Helper: posture decision
def posture_from_angles(avg_angle, neck_angle):
    # thresholds tuned for webcam portrait near-screen; adjust if needed
    if avg_angle > 155 and 140 < neck_angle < 175:
        return "Good Posture"
    if avg_angle >= 140:
        return "Mild Lean"
    return "Bad Posture"

# Video transformer: receives frames, returns annotated frames
class Transformer(VideoTransformerBase):
    def _init_(self):
        self.angle_buffer = []
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # MediaPipe pose
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose_model.process(rgb)

        avg_angle = 0.0
        neck_angle = 0.0
        posture_label = "No person"
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # use left and right sides and average
            def to_xy(l):
                return (int(l.x * w), int(l.y * h))

            # choose shoulders, hips, knees, nose
            ls = to_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            rs = to_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            lh = to_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
            rh = to_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
            lk = to_xy(lm[mp_pose.PoseLandmark.LEFT_KNEE.value])
            rk = to_xy(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            nose = to_xy(lm[mp_pose.PoseLandmark.NOSE.value])

            # compute angles for left & right (shoulder-hip-knee)
            def angle(a, b, c):
                a = np.array(a); b = np.array(b); c = np.array(c)
                radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                ang = abs(radians * 180.0 / np.pi)
                if ang > 180:
                    ang = 360 - ang
                return ang

            ang_l = angle(ls, lh, lk)
            ang_r = angle(rs, rh, rk)
            avg_angle = float((ang_l + ang_r) / 2.0)

            # neck angle: nose - shoulder - hip (averaged)
            neck_l = angle(nose, ls, lh)
            neck_r = angle(nose, rs, rh)
            neck_angle = float((neck_l + neck_r) / 2.0)

            # smoothing
            with self.lock:
                self.angle_buffer.append(avg_angle)
                if len(self.angle_buffer) > ANGLE_BUFFER_LEN:
                    self.angle_buffer.pop(0)
                smoothed = float(np.mean(self.angle_buffer))

            posture_label = posture_from_angles(smoothed, neck_angle)

            # draw landmarks
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # annotate
            cv2.putText(img, f"Angle: {int(smoothed)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(img, f"Neck: {int(neck_angle)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
            cv2.putText(img, f"Posture: {posture_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,180,255), 2)

        # Emotion detection (single face)
        try:
            em = emotion_detector.detect_emotions(img)
            if em:
                # emotions dict
                em0 = em[0]["emotions"]
                # dominant and confidence
                dominant = max(em0, key=em0.get)
                confidence = float(em0[dominant])
            else:
                dominant = "neutral"
                confidence = 0.0
        except Exception:
            dominant = "neutral"
            confidence = 0.0

        # Put emotion text
        cv2.putText(img, f"Emotion: {dominant} ({confidence:.2f})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

        # push a row to session queue for main thread to consume
        try:
            row = {
                "timestamp": datetime.now(),
                "posture": posture_label,
                "angle": int(avg_angle),
                "emotion": dominant,
                "confidence": round(confidence, 2),
            }
            # non-blocking put
            if data_q.qsize() < 2000:
                data_q.put_nowait(row)
        except Exception:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Start or restart session
if start_button:
    st.session_state.session_df = pd.DataFrame(columns=["timestamp", "posture", "angle", "emotion", "confidence"])
    # clear queue
    while not data_q.empty():
        try: data_q.get_nowait()
        except: break

# Start WebRTC streamer with our transformer class
webrtc_ctx = webrtc_streamer(
    key="ai-posture-demo",
    video_transformer_factory=Transformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)


# Main loop: read queued data and update UI charts
def update_ui_from_queue():
    updated = False
    while not data_q.empty():
        try:
            row = data_q.get_nowait()
        except queue.Empty:
            break
        st.session_state.session_df = pd.concat([st.session_state.session_df, pd.DataFrame([row])], ignore_index=True)
        updated = True
    if updated:
        # Metrics
        last = st.session_state.session_df.iloc[-1]
        posture_metric.metric("Posture", last["posture"])
        angle_metric.metric("Angle (deg)", int(last["angle"]))
        emotion_metric.metric("Emotion", last["emotion"])
        confidence_metric.metric("Confidence", float(last["confidence"]))

        # Charts
        df = st.session_state.session_df
        posture_counts = df["posture"].value_counts()
        posture_chart.bar_chart(posture_counts)

        emotion_counts = df["emotion"].value_counts()
        emotion_chart.pyplot(plt.figure(figsize=(3,3)))
        fig, ax = plt.subplots()
        emotion_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

        # last 10 rows table
        log_table.dataframe(df.tail(10))

# Periodically update UI while the webrtc component runs
if webrtc_ctx.state.playing:
    # poll data queue and update UI
    update_ui_from_queue()

# Download CSV
if download_button:
    csv = st.session_state.session_df.to_csv(index=False)
    st.download_button("Download session CSV", csv, file_name=f"{session_name}_session_log.csv", mime="text/csv")

st.markdown("---")
st.caption("Notes: run this page locally. If webcam fails, ensure browser allows camera access and close other apps using camera.")
