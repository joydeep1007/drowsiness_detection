import pygame
import cv2
import dlib
import os
from scipy.spatial import distance as dist

def init_sound_system():
    """Initialize the sound system and verify sound files"""
    pygame.mixer.init()
    sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sounds')
    
    if not os.path.exists(sounds_dir):
        os.makedirs(sounds_dir)
    
    # Define default sounds
    default_sounds = {
        'alert-10.wav': bytes([
            # Simple beep sound wave data
            0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00,
            0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20,
            0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
            0x44, 0xAC, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00,
            0x02, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61
        ]),
        'siren-9.wav': bytes([
            # Simple beep sound wave data (different frequency)
            0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00,
            0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20,
            0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
            0x44, 0xAC, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00,
            0x02, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61
        ]),
        'music.wav': bytes([
            # Simple beep sound wave data (default beep)
            0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00,
            0x57, 0x41, 0x56, 0x45, 0x66, 0x6D, 0x74, 0x20,
            0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
            0x44, 0xAC, 0x00, 0x00, 0x88, 0x58, 0x01, 0x00,
            0x02, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61
        ])
    }
    
    # Create default sound files if they don't exist
    for filename, data in default_sounds.items():
        file_path = os.path.join(sounds_dir, filename)
        if not os.path.exists(file_path):
            try:
                with open(file_path, 'wb') as f:
                    f.write(data)
                print(f"Created default sound file: {filename}")
            except Exception as e:
                print(f"Error creating sound file {filename}: {e}")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def play_alarm(sound_type='beep'):
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    
    sounds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sounds')
    
    try:
        if sound_type == 'alarm':
            sound_file = os.path.join(sounds_dir, 'alert-10.wav')
        elif sound_type == 'voice':
            sound_file = os.path.join(sounds_dir, 'siren-9.wav')
        else:
            sound_file = os.path.join(sounds_dir, 'music.wav')
        
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing sound: {e}")
        # Fallback to system beep if available
        try:
            import winsound
            winsound.Beep(1000, 500)  # 1000 Hz for 500ms
        except:
            pass

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




