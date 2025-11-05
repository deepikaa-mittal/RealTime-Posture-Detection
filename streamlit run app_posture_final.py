# app_posture_final.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

st.set_page_config(page_title="Posture Detector — Final", layout="wide")
st.title("📸 Real-time Posture Detection (Improved Accuracy)")
st.markdown("Sit naturally in front of the webcam. The app computes a posture score (0–100) based on spine and neck alignment.")

# -------------------------
# Controls
# -------------------------
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    run = st.checkbox("Start webcam")
with col2:
    camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
with col3:
    smoothing_k = st.slider("Smoothing frames (higher = smoother)", 1, 30, 8)

st.info("Tip: Close other apps using the camera (Zoom/Teams). Sit ~1-2 meters away from camera for best detection.")

# UI placeholders
video_placeholder = st.image([])    # video frame
score_placeholder = st.empty()      # numerical score and posture label
advice_placeholder = st.empty()
metrics_col1, metrics_col2 = st.columns(2)
spine_placeholder = metrics_col1.empty()
neck_placeholder = metrics_col2.empty()

# Footer (developer credit will be shown always)
footer = st.empty()

# Mediapipe setup (create once)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)

# smoothing buffers
spine_buf = []
neck_buf = []
time_last = 0.0

def angle_between_points(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.dot(ba, bc) / denom
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return np.degrees(np.arccos(cosang))

def midpoint(p1, p2):
    return ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)

def compute_posture_score(spine_angle, neck_angle):
    """
    Compute a composite posture score (0-100).
    Ideal angles close to 180 degrees. Penalize deviations.
    We weight spine 0.7 and neck 0.3.
    """
    # clamp angles to [0, 180]
    spine = float(np.clip(spine_angle, 0.0, 180.0))
    neck = float(np.clip(neck_angle, 0.0, 180.0))

    # deviations from ideal (180)
    dev_spine = abs(180.0 - spine)
    dev_neck = abs(180.0 - neck)

    # map deviation to 0-100 where 0 means huge deviation (>=60 deg), 100 means perfect
    def score_from_dev(dev, max_dev=60.0):
        s = max(0.0, 100.0 * (1.0 - (dev / max_dev)))
        return s

    s_spine = score_from_dev(dev_spine)
    s_neck = score_from_dev(dev_neck)

    # weighted combination
    score = 0.7 * s_spine + 0.3 * s_neck
    return float(np.clip(score, 0.0, 100.0)), s_spine, s_neck

# open camera lazily (only when run is True)
cap = None
try:
    while run:
        if cap is None:
            cap = cv2.VideoCapture(int(camera_index))
            time.sleep(0.4)  # give camera a moment to warm up

        ret, frame = cap.read()
        if not ret or frame is None:
            st.warning("⚠️ Couldn't read from webcam. Make sure it's free and the index is correct.")
            break

        # flip for mirror-view
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # mediapipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        posture_label = "No person detected"
        posture_score = 0.0
        spine_angle = 0.0
        neck_angle = 0.0

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # helper to convert to pixel coords
            def to_xy(lm_pt):
                return (lm_pt.x * w, lm_pt.y * h)

            # key points needed:
            left_shoulder = to_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            right_shoulder = to_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            left_hip = to_xy(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
            right_hip = to_xy(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
            left_knee = to_xy(lm[mp_pose.PoseLandmark.LEFT_KNEE.value])
            right_knee = to_xy(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            nose = to_xy(lm[mp_pose.PoseLandmark.NOSE.value])
            left_ear = to_xy(lm[mp_pose.PoseLandmark.LEFT_EAR.value])
            right_ear = to_xy(lm[mp_pose.PoseLandmark.RIGHT_EAR.value])

            shoulders_mid = midpoint(left_shoulder, right_shoulder)
            hips_mid = midpoint(left_hip, right_hip)
            knees_mid = midpoint(left_knee, right_knee)
            ears_mid = midpoint(left_ear, right_ear)

            # compute angles (spine: shoulders_mid - hips_mid - knees_mid)
            try:
                spine_angle = angle_between_points(shoulders_mid, hips_mid, knees_mid)
                # neck: nose - shoulders_mid - hips_mid (how head sits on shoulders)
                neck_angle = angle_between_points(nose, shoulders_mid, hips_mid)
            except Exception:
                spine_angle = 0.0
                neck_angle = 0.0

            # smoothing buffers
            spine_buf.append(spine_angle)
            neck_buf.append(neck_angle)
            if len(spine_buf) > smoothing_k:
                spine_buf.pop(0)
            if len(neck_buf) > smoothing_k:
                neck_buf.pop(0)

            spine_s = float(np.mean(spine_buf))
            neck_s = float(np.mean(neck_buf))

            posture_score, s_spine, s_neck = compute_posture_score(spine_s, neck_s)

            # label
            if posture_score >= 80:
                posture_label = "Excellent"
            elif posture_score >= 65:
                posture_label = "Good"
            elif posture_score >= 45:
                posture_label = "Leaning"
            else:
                posture_label = "Poor"

            # draw landmarks nicely
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1))

            # overlay angles & score on frame
            cv2.putText(frame, f"Score: {int(posture_score)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 200, 10), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Posture: {posture_label}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 180, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Spine: {int(spine_s)} deg", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Neck: {int(neck_s)} deg", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        else:
            # small hint text when no person found
            cv2.putText(frame, "No person detected — move into the camera view", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)

        # display frame
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # display metrics in the side placeholders
        if results.pose_landmarks:
            score_placeholder.markdown(f"### Posture score: **{int(posture_score)} / 100** — {posture_label}")
            spine_placeholder.metric("Spine angle (smoothed)", f"{int(np.mean(spine_buf))}°")
            neck_placeholder.metric("Neck angle (smoothed)", f"{int(np.mean(neck_buf))}°")
            # simple advice
            if posture_label in ["Poor", "Leaning"]:
                advice_placeholder.warning("⚠️ Try sitting upright: roll shoulders back, align head over shoulders.")
            else:
                advice_placeholder.success("✅ Good posture — keep it up!")
        else:
            score_placeholder.markdown("### Posture score: —")
            spine_placeholder.metric("Spine angle (smoothed)", "—")
            neck_placeholder.metric("Neck angle (smoothed)", "—")
            advice_placeholder.info("Position yourself in front of the webcam.")

        # throttle loop a little to reduce CPU + prevent freeze
        time.sleep(0.03)

    # if while run ends (user unchecked the checkbox), tidy up
    if cap is not None:
        cap.release()
        cap = None

except Exception as e:
    st.error(f"Unexpected error: {e}")
    if cap is not None:
        cap.release()

finally:
    footer.markdown(
        "<hr><div style='text-align:center;color:gray'>Developed by Deepika Mittal and Avish Sharma | VIT Vellore</div>",
        unsafe_allow_html=True
    )
