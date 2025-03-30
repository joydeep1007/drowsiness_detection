# Drowsiness Detection System

A real-time drowsiness detection system built with Python, OpenCV, and dlib. This web application monitors user alertness through webcam feed and provides timely alerts to prevent drowsy driving.

## Features

- Real-time eye tracking
- Drowsiness detection using Eye Aspect Ratio (EAR)
- Audio alerts when drowsiness is detected
- Web interface with live video feed
- Dark/Light theme toggle
- Real-time EAR value display

## Prerequisites

- Python 3.8+
- Webcam
- Windows/Linux/MacOS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/drowsiness_detection.git
cd drowsiness_detection
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/MacOS
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the shape predictor file:
- Download [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- Extract and place it in the project root directory

## Project Structure

```
drowsiness_detection/
├── app.py                     # Flask application
├── drowsiness_detection.py    # Core detection logic
├── static/
│   └── css/
│       └── style.css         # Stylesheet
├── templates/
│   └── index.html            # Web interface
├── shape_predictor_68_face_landmarks.dat
├── music.wav                 # Alert sound
└── requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

3. Grant camera permissions when prompted
4. The system will start monitoring for signs of drowsiness

## How it Works

1. The system captures video feed from your webcam
2. Facial landmarks are detected using dlib
3. Eye Aspect Ratio (EAR) is calculated for both eyes
4. If EAR falls below threshold for several consecutive frames:
   - Visual alert is displayed
   - Audio alarm is triggered

## Configuration

Adjust these parameters in `drowsiness_detection.py`:
```python
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
FRAME_THRESHOLD = 20  # Consecutive frames for alert
```
