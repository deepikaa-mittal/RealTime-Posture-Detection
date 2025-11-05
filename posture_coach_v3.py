import cv2
import mediapipe as mp
import numpy as np
from fer import FER
import pyttsx3
import matplotlib.pyplot as plt

# Initialize MediaPipe & FER
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
emotion_detector = FER(mtcnn=True)

# Voice Engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 0.9)

# Tracking variables
good_posture_frames = 0
bad_posture_frames = 0
emotion_log = {"happy": 0, "neutral": 0, "sad": 0, "angry": 0, "surprise": 0}
posture_score = 100

# Camera
cap = cv2.VideoCapture(0)
plt.ion()
fig, ax = plt.subplots()
ax.set_title("Posture Accuracy (%)")
line, = ax.plot([], [], 'g-')
scores = []

def speak(text):
    engine.say(text)
    engine.runAndWait()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Pose detection
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Key points
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]

        # Neck and back angles
        angle = calculate_angle(shoulder, hip, knee)
        neck_angle = calculate_angle(
            [landmarks[mp_pose.PoseLandmark.NOSE.value].x * w,
             landmarks[mp_pose.PoseLandmark.NOSE.value].y * h],
            shoulder,
            hip
        )

        # Posture logic
        if 150 < angle < 180 and 140 < neck_angle < 175:
            good_posture_frames += 1
            bad_posture_frames = max(bad_posture_frames - 1, 0)
            posture_status = "Good Posture ✅"
            color = (0, 255, 0)
        else:
            bad_posture_frames += 1
            good_posture_frames = max(good_posture_frames - 1, 0)
            posture_status = "Bad Posture ⚠️"
            color = (0, 0, 255)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, posture_status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Calculate posture accuracy score
        total = good_posture_frames + bad_posture_frames + 1
        posture_score = int((good_posture_frames / total) * 100)

    # Emotion detection
    emotion = emotion_detector.detect_emotions(frame)
    if emotion:
        dominant = max(emotion[0]["emotions"], key=emotion[0]["emotions"].get)
        if dominant in emotion_log:
            emotion_log[dominant] += 1
        cv2.putText(frame, f"Emotion: {dominant}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("AI Posture & Emotion Coach", frame)

    # Update dashboard chart
    scores.append(posture_score)
    if len(scores) > 30:
        scores.pop(0)
    line.set_xdata(np.arange(len(scores)))
    line.set_ydata(scores)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)
    fig.canvas.draw()

    # Voice feedback occasionally
    if bad_posture_frames > 30 and bad_posture_frames % 50 == 0:
        speak("Your posture looks incorrect, please sit straight.")
    elif good_posture_frames > 50 and good_posture_frames % 100 == 0:
        speak("Good posture maintained, keep it up!")

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
