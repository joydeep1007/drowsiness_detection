import os
from flask import Flask, render_template, Response, send_from_directory, jsonify, request
import cv2
import dlib
import pygame
from scipy.spatial import distance as dist
from drowsiness_detection import eye_aspect_ratio, play_alarm, init_sound_system
from detection_state import DetectionState

# Update the static folder path to be absolute
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__, static_folder=static_folder)

# Initialize pygame before sound system
pygame.init()
init_sound_system()

# Add settings storage
app_settings = {
    'alertSound': 'beep',
    'volume': 80,
    'earThreshold': 0.25,
    'frameThreshold': 20,
    'alertMessage': 'DROWSINESS ALERT!'  # Add this line
}

# Add global detection state
detection_active = False

# Initialize detection state
detection_state = DetectionState()

# Add this route for debugging static files
@app.route('/static/<path:filename>')
def custom_static(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        print(f"Error serving static file: {e}")
        return str(e), 404

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_active
    state = request.json.get('state', False)
    detection_active = state
    return jsonify({'status': 'success', 'detection_active': detection_active})

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def generate_frames():
    global app_settings, detection_active
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            if detection_active:
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
                        
                        # Updated drowsiness detection using DetectionState
                        is_drowsy = detection_state.update(ear, float(app_settings['earThreshold']))
                        
                        if is_drowsy and detection_state.frame_count >= int(app_settings['frameThreshold']):
                            cv2.putText(frame, app_settings['alertMessage'], (50, 100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                            play_alarm(app_settings['alertSound'])
                        
                        # Add statistics overlay
                        stats = detection_state.get_statistics()
                        cv2.putText(frame, f"EAR: {ear:.2f} | Alerts: {stats['total_alerts']}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"Error in face detection: {e}")
                    continue
            else:
                # Just display the frame without detection
                detection_state.reset()
            
            # Convert frame to bytes for streaming
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error encoding frame: {e}")
                continue
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    try:
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video feed: {e}")
        return "Video feed error", 500

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

@app.route('/statistics')
def statistics_page():
    return render_template('statistics.html')

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    global app_settings
    if request.method == 'POST':
        app_settings.update(request.json)
        return jsonify({'status': 'success'})
    return jsonify(app_settings)

@app.route('/api/stats')
def get_stats():
    global detection_state
    return jsonify(detection_state.get_statistics())

if __name__ == '__main__':
    print(f"Static folder path: {app.static_folder}")
    app.run(debug=True)