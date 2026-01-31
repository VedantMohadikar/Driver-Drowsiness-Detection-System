import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Drawing utility
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Thresholds
EYE_CLOSED_FRAMES = 20
YAWN_THRESHOLD = 25

eye_counter = 0

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Eye landmarks (approx)
            left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]

            def eye_aspect_ratio(eye_points):
                p1 = euclidean_distance(eye_points[1], eye_points[5])
                p2 = euclidean_distance(eye_points[2], eye_points[4])
                p3 = euclidean_distance(eye_points[0], eye_points[3])
                return (p1 + p2) / (2.0 * p3)

            left_eye_pts = [(int(face_landmarks.landmark[i].x * w),
                              int(face_landmarks.landmark[i].y * h)) for i in left_eye]

            right_eye_pts = [(int(face_landmarks.landmark[i].x * w),
                               int(face_landmarks.landmark[i].y * h)) for i in right_eye]

            ear = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2

            # Mouth landmarks for yawning
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            mouth_opening = abs(upper_lip.y - lower_lip.y) * h

            # Logic
            if ear < 0.2:
                eye_counter += 1
                if eye_counter > EYE_CLOSED_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                eye_counter = 0

            if mouth_opening > YAWN_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
