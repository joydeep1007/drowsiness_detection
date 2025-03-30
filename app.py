import os
from flask import Flask, render_template, Response, send_from_directory
import cv2
import dlib
from scipy.spatial import distance as dist
import pygame
from drowsiness_detection import eye_aspect_ratio, play_alarm

# Update the static folder path to be absolute
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__, static_folder=static_folder)

# Add this route for debugging static files
@app.route('/static/<path:filename>')
def custom_static(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        print(f"Error serving static file: {e}")
        return str(e), 404

# Initialize pygame for sound
pygame.mixer.init()
pygame.mixer.music.load("music.wav")

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants
EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20
frame_count = 0

def generate_frames():
    global frame_count
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if gray.dtype != 'uint8':
                gray = gray.astype('uint8')
            
            try:
                faces = detector(gray)
                
                for face in faces:
                    landmarks = predictor(gray, face)
                    
                    left_eye = []
                    right_eye = []
                    
                    # Left eye landmarks
                    for n in range(36, 42):
                        x, y = landmarks.part(n).x, landmarks.part(n).y
                        left_eye.append((x, y))
                    
                    # Right eye landmarks
                    for n in range(42, 48):
                        x, y = landmarks.part(n).x, landmarks.part(n).y
                        right_eye.append((x, y))
                    
                    # Calculate EAR for both eyes
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    
                    # Draw landmarks on eyes
                    for (x, y) in left_eye + right_eye:
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Drowsiness detection logic
                    if ear < EAR_THRESHOLD:
                        frame_count += 1
                        if frame_count >= FRAME_THRESHOLD:
                            cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                            play_alarm()
                    else:
                        frame_count = 0
                        
                    # Display EAR value
                    cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            except Exception as e:
                print(f"Error in face detection: {e}")
                continue
            
            # Convert frame to bytes for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"Static folder path: {app.static_folder}")
    app.run(debug=True)