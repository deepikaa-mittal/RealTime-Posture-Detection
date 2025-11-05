import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

# Initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Open webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect pose
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # Calculate angle
            angle = calculate_angle(ear, shoulder, hip)

            # Classify posture
            if angle > 160:
                posture_status = "Good Posture"
            elif 130 < angle <= 160:
                posture_status = "Slight Lean"
            else:
                posture_status = "Bad Posture"

        except:
            posture_status = "No person detected"
            angle = 0

        # ---------- EMOTION DETECTION -------------
        try:
            # Analyze one face per frame
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion'].capitalize()
        except:
            dominant_emotion = "Unknown"

        # ---------- FEEDBACK LOGIC -------------
        feedback = ""
        if posture_status == "Good Posture":
            if dominant_emotion in ["Happy", "Neutral"]:
                feedback = "Excellent! You look confident and comfortable 😄"
            elif dominant_emotion in ["Sad", "Tired", "Disgust"]:
                feedback = "Good posture, but you look tired — relax a bit 💆‍♀️"
        elif posture_status == "Slight Lean":
            feedback = "Straighten up a little! You're almost there 💪"
        elif posture_status == "Bad Posture":
            if dominant_emotion in ["Sad", "Tired"]:
                feedback = "Bad posture + low mood detected. Take a short walk 🌿"
            else:
                feedback = "Watch your posture — align your back and neck 🧘‍♀️"

        # ---------- DISPLAY OUTPUT -------------
        cv2.putText(image, f"Posture: {posture_status}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(image, f"Angle: {int(angle)} deg", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(image, f"Emotion: {dominant_emotion}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(image, f"Feedback: {feedback}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("AI-Based Emotion-Aware Posture Coach", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
