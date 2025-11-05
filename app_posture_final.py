import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="AI Posture Detector", layout="wide")
st.title("🧍‍♀️ Real-Time Posture Detection (Accurate)")

st.markdown("""
This app uses your webcam to measure **spine** and **neck alignment** in real-time using Mediapipe Pose.
It gives you a **Posture Score (0–100)** and live feedback.
""")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("🎛️ Controls")
    run = st.checkbox("Start Webcam", value=False)
    st.markdown("---")
    st.info("💡 Sit straight for 3–5 seconds after starting to calibrate your best posture.")
    st.markdown(
        "<hr><center>Developed by <b>Deepika Mittal</b> and <b>Avish Sharma</b><br>VIT Vellore</center>",
        unsafe_allow_html=True
    )

# -----------------------------
# Layout placeholders
# -----------------------------
frame_placeholder = st.empty()
score_placeholder = st.empty()
col1, col2 = st.columns(2)
spine_placeholder = col1.empty()
neck_placeholder = col2.empty()
feedback_placeholder = st.empty()

# -----------------------------
# Mediapipe setup
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# -----------------------------
# Helper functions
# -----------------------------
def angle_between(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def compute_score(spine, neck):
    dev_spine = abs(spine)
    dev_neck = abs(neck)
    s_spine = max(0, 100 - (dev_spine * 1.5))
    s_neck = max(0, 100 - (dev_neck * 2))
    return float(0.7 * s_spine + 0.3 * s_neck)

# -----------------------------
# Start webcam loop
# -----------------------------
if run:
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    stframe = st.empty()
    posture_ref = {"spine": None, "neck": None}

    while run:
        success, frame = cap.read()
        if not success:
            st.warning("⚠️ Could not access webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        spine_angle, neck_angle, score = 0, 0, 0
        label = "No person detected"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            def xy(id): return (lm[id].x * w, lm[id].y * h)
            left_shoulder, right_shoulder = xy(mp_pose.PoseLandmark.LEFT_SHOULDER), xy(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            left_hip, right_hip = xy(mp_pose.PoseLandmark.LEFT_HIP), xy(mp_pose.PoseLandmark.RIGHT_HIP)
            nose = xy(mp_pose.PoseLandmark.NOSE)

            mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2,
                            (left_shoulder[1] + right_shoulder[1]) / 2)
            mid_hip = ((left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2)
            vertical = np.array([0, -1])

            spine_vec = np.array([mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1]])
            neck_vec = np.array([nose[0] - mid_shoulder[0], nose[1] - mid_shoulder[1]])

            spine_angle = angle_between(spine_vec, vertical)
            neck_angle = angle_between(neck_vec, vertical)

            # calibration
            if posture_ref["spine"] is None:
                posture_ref["spine"] = spine_angle
                posture_ref["neck"] = neck_angle

            spine_dev = spine_angle - posture_ref["spine"]
            neck_dev = neck_angle - posture_ref["neck"]

            score = compute_score(spine_dev, neck_dev)

            if score > 85:
                label = "Excellent"
                color = (0, 255, 0)
            elif score > 70:
                label = "Good"
                color = (0, 200, 255)
            elif score > 50:
                label = "Leaning"
                color = (0, 165, 255)
            else:
                label = "Poor"
                color = (0, 0, 255)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Posture: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Spine: {spine_dev:.1f}°  Neck: {neck_dev:.1f}°", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Score: {int(score)}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2)

        # display frame
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            score_placeholder.markdown(f"### 🟩 **Posture Score: {int(score)} / 100 — {label}**")
            spine_placeholder.metric("Spine Deviation (°)", f"{abs(spine_angle - posture_ref['spine']):.1f}")
            neck_placeholder.metric("Neck Deviation (°)", f"{abs(neck_angle - posture_ref['neck']):.1f}")

            if label in ["Poor", "Leaning"]:
                feedback_placeholder.warning("⚠️ Try straightening your back and keeping your neck aligned.")
            else:
                feedback_placeholder.success("✅ Great posture! Keep maintaining it.")
        else:
            score_placeholder.markdown("### Waiting for detection...")

        time.sleep(0.03)

    cap.release()

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "<hr><center><b>Developed by Deepika Mittal and Avish Sharma | VIT Vellore</b></center>",
    unsafe_allow_html=True
)
