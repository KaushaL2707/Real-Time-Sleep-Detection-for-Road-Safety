import cv2
import mediapipe as mp
import numpy as np
import time
from pygame import mixer
import pygame

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate EAR
def calculate_ear(landmarks, eye_points):
    # Eye landmarks
    a = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
    b = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
    c = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
    ear = (a + b) / (2.0 * c)
    return ear

# Indices for left and right eyes (Mediapipe specific)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Threshold for closed eyes
EAR_THRESHOLD = 0.2

count=0
score=0

# Initialize Pygame
pygame.init()

# Initialize the mixer module
mixer.init()

sound = mixer.Sound('mixkit-alert-alarm-1005.wav')


# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    height,width = frame.shape[:2]
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with Mediapipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmark coordinates
            landmarks = []
            for lm in face_landmarks.landmark:
                ih, iw, _ = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                landmarks.append((x, y))

            # Calculate EAR for both eyes
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)

            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0

            # Highlight the eyes with custom shapes
            for eye_points in [LEFT_EYE, RIGHT_EYE]:
                # Get bounding box around the eye
                x_min = min([landmarks[pt][0] for pt in eye_points])
                y_min = min([landmarks[pt][1] for pt in eye_points])
                x_max = max([landmarks[pt][0] for pt in eye_points])
                y_max = max([landmarks[pt][1] for pt in eye_points])

                # Draw a rounded rectangle around the eye
                #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                cv2.circle(frame, ((x_min + x_max) // 2, (y_min + y_max) // 2), 20, (255, 0, 0), 2)

            # Check if eyes are closed
            if avg_ear < EAR_THRESHOLD:
                score = score+1
                cv2.putText(frame, "Eyes Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                score=score-1
                cv2.putText(frame, "Eyes Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if(score<0):
                score=0
            cv2.putText(frame,'Score:'+str(score),(100,height-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)

            #300 for 10 seconds
            if(score>300):
                sound.play()

    # Display the frame
    cv2.imshow('Eye State Detection', frame)
    time.sleep(0)

    # Exit loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
