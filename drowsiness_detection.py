
import pygame
import cv2
import dlib
from scipy.spatial import distance as dist

pygame.mixer.init()
pygame.mixer.music.load("music.wav")

# Define Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def play_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    pygame.mixer.music.play()
    

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# EAR threshold and frame count for drowsiness detection
EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20
frame_count = 0

def detect_drowsiness(frame):
    global frame_count

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get eye coordinates
        left_eye = []
        right_eye = []
        
        # Left eye points (36-41)
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
        # Right eye points (42-47)
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
        # Calculate EAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Check for drowsiness
        if ear < EAR_THRESHOLD:
            frame_count += 1
            if frame_count >= FRAME_THRESHOLD:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                play_alarm()
        else:
            frame_count = 0
            
        # Display EAR value
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame


    

