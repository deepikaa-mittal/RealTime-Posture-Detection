import cv2
import mediapipe as mp
import numpy as np
from fer import FER
import pyttsx3

# Initialize components
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
emotion_detector = FER()
engine = pyttsx3.init()

# Voice setup
engine.setProperty("rate", 170)
engine.setProperty("volume", 0.9)

# Start camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera not detected.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate shoulder slope for posture
        shoulder_angle = np.degrees(np.arctan2(
            right_shoulder.y - left_shoulder.y,
            right_shoulder.x - left_shoulder.x
        ))

        # Emotion detection
        emotion, score = None, None
        try:
            emotion_data = emotion_detector.detect_emotions(frame)
            if emotion_data:
                top_emotion = max(emotion_data[0]["emotions"], key=emotion_data[0]["emotions"].get)
                emotion = top_emotion
                score = emotion_data[0]["emotions"][top_emotion]
        except Exception:
            pass

        # Evaluate posture & emotion
        feedback = ""
        if abs(shoulder_angle) > 10:
            feedback = "Your posture seems tilted. Sit straight!"
        else:
            feedback = "Good posture maintained!"

        if emotion:
            feedback += f" You seem {emotion}."

        # Display on screen
        cv2.putText(frame, feedback, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.2f}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        cv2.imshow('AI-Based Emotion-Aware Posture Coach', frame)

        # Optional voice feedback
        if feedback:
            engine.say(feedback)
            engine.runAndWait()

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to close
        break

cap.release()
cv2.destroyAllWindows()
