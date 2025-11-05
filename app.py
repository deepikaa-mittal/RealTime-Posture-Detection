import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.set_page_config(page_title="AI Posture Detector", layout="wide")
st.title("🧍‍♀️ Real-Time Posture Detection (Accurate)")

st.markdown(
    "This app measures your spine and neck alignment in real time using Mediapipe Pose. "
    "It gives you a posture score between 0–100."
)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")
    run = st.checkbox("Start webcam")
    smooth_factor = st.slider("Smoothing frames", 1, 30, 10)
    st.markdown("---")
    st.info("💡 Tip: Sit upright for 3–5 seconds after starting to calibrate your good posture.")
    st.markdown("<hr><center>Developed by <b>Deepika Mittal</b> and <b>Avish Sharma</b><br>VIT Vellore</center>", unsafe_allow_html=True)

# -----------------------------
# Setup placeholders
# -----------------------------
frame_window = st.image([])
score_display = st.empty()
advice_display = st.empty()
angle_col1, angle_col2 = st.columns(2)
spine_display = angle_col1.empty()
neck_display = angle_col2.empty()

# -----------------------------
# Mediapipe setup
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)

# -----------------------------
# Helper functions
# -----------------------------
def vector_angle(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    cosang = np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def moving_avg(buffer, val, k):
    buffer.append(val)
    if len(buffer) > k:
        buffer.pop(0)
    return np.mean(buffer)

def compute_score(spine, neck):
    # ideal: spine ≈ 0°, neck ≈ 0° relative to vertical
    dev_spine = abs(spine)
    dev_neck = abs(neck)
    s_spine = max(0, 100 - (dev_spine * 1.2))
    s_neck = max(0, 100 - (dev_neck * 1.5))
    return float(0.7 * s_spine + 0.3 * s_neck)

# -----------------------------
# Buffers for smoothing
# -----------------------------
spine_buf, neck_buf = [], []
spine_ref, neck_ref = None, None  # calibration references

cap = None
try:
    while run:
        if cap is None:
            cap = cv2.VideoCapture(0)
            time.sleep(0.5)
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not found. Try another camera or close other apps using it.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        posture_score = 0
        spine_angle = 0
        neck_angle = 0
        label = "No person detected"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            def xy(id): return (lm[id].x * w, lm[id].y * h)
            left_shoulder = xy(mp_pose.PoseLandmark.LEFT_SHOULDER)
            right_shoulder = xy(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            left_hip = xy(mp_pose.PoseLandmark.LEFT_HIP)
            right_hip = xy(mp_pose.PoseLandmark.RIGHT_HIP)
            nose = xy(mp_pose.PoseLandmark.NOSE)

            mid_shoulder = ((left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2)
            mid_hip = ((left_hip[0]+right_hip[0])/2, (left_hip[1]+right_hip[1])/2)
            vertical = np.array([0, -1])

            # Spine vector (hip to shoulder)
            spine_vec = np.array([mid_shoulder[0]-mid_hip[0], mid_shoulder[1]-mid_hip[1]])
            spine_angle = vector_angle(spine_vec, vertical)

            # Neck vector (shoulder to nose)
            neck_vec = np.array([nose[0]-mid_shoulder[0], nose[1]-mid_shoulder[1]])
            neck_angle = vector_angle(neck_vec, vertical)

            # calibration
            if spine_ref is None:
                spine_ref, neck_ref = spine_angle, neck_angle

            # relative deviation
            spine_rel = spine_angle - spine_ref
            neck_rel = neck_angle - neck_ref

            # smoothing
            sm_spine = moving_avg(spine_buf, spine_rel, smooth_factor)
            sm_neck = moving_avg(neck_buf, neck_rel, smooth_factor)

            # score computation
            posture_score = compute_score(sm_spine, sm_neck)

            # label based on score
            if posture_score >= 85:
                label = "Excellent"
            elif posture_score >= 70:
                label = "Good"
            elif posture_score >= 50:
                label = "Leaning"
            else:
                label = "Poor"

            # draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Posture: {label} ({int(posture_score)})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 200, 10), 2)
            cv2.putText(frame, f"Spine: {sm_spine:.1f}° Neck: {sm_neck:.1f}°", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 0), 2)
        else:
            cv2.putText(frame, "No person detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # update UI
        if results.pose_landmarks:
            score_display.markdown(f"### 🟩 Posture Score: **{int(posture_score)} / 100** — *{label}*")
            spine_display.metric("Spine Deviation (°)", f"{abs(spine_angle - spine_ref):.1f}")
            neck_display.metric("Neck Deviation (°)", f"{abs(neck_angle - neck_ref):.1f}")

            if label in ["Poor", "Leaning"]:
                advice_display.warning("⚠️ Try straightening your back and keeping your chin level.")
            else:
                advice_display.success("✅ Great posture! Keep maintaining it.")
        else:
            score_display.markdown("### Waiting for detection...")

        time.sleep(0.03)

    if cap:
        cap.release()

finally:
    st.markdown("<hr><center><b>Developed by Deepika Mittal and Avish Sharma | VIT Vellore</b></center>", unsafe_allow_html=True)
