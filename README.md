# Face Emotion Detection

Small Flask + Socket.IO application that performs real-time face emotion detection using a prebuilt Keras model and OpenCV face detection. This project demonstrates streaming webcam frames to a browser, running emotion predictions on detected faces, and emitting live emotion data via WebSockets.

## Overview

- Real-time video captured from a webcam is processed by OpenCV to detect faces.
- A prebuilt Keras model (`src/emotion_model.h5`) predicts emotion probabilities for each detected face.
- The app serves a video stream and emits JSON emotion updates via Socket.IO for dashboard visualization.

## Prebuilt models & data

- Emotion classification model: `src/emotion_model.h5` (pretrained Keras HDF5 model). Place this file in `src/` (already included in this repo).

- Face detector: `src/haarcascade_frontalface_default.xml` (OpenCV Haar cascade). Also included in `src/`.

These are prebuilt assets bundled with the project; you can replace them with your own models if desired.

## Project layout

- `src/` — model and face cascade, and helper scripts.
- `webapp/` — Flask application and frontend templates.
  - `webapp/app.py` — main Flask + Socket.IO server.
  - `webapp/templates/` — HTML templates for home, login, dashboard.
  - `webapp/static/` — CSS, JS and `captures/` saved images.

## Requirements

Python 3.8+ recommended. Key packages:

- `flask`
- `flask-socketio`
- `tensorflow` (or `tensorflow-cpu` depending on your environment)
- `opencv-python`
- `numpy`

I can add a `requirements.txt` for you — let me know.

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
# or if you don't have requirements.txt:
python -m pip install flask flask-socketio tensorflow opencv-python numpy
```

## Run the app

From the `webapp` directory run:

```powershell
python .\app.py
```

Open your browser at `http://localhost:5000` and go to the dashboard to see the live video and emotion updates.

Notes:
- On Windows PowerShell you may need to run `.\app.py` or `python app.py` — `python app.py` is recommended.
- The app uses your default webcam (index 0). If you have multiple cameras, change the index in `app.py` where `cv2.VideoCapture(0)` is called.
- Socket.IO is used to broadcast emotion percentages in real time to clients.

## Endpoints & features

- `/` — Home page
- `/dashboard` — Dashboard (requires a simple session login)
- `/video_feed` — MJPEG stream of the webcam frames
- `/get_emotions` — Returns latest emotion percentages as JSON
- `/capture` — POST endpoint that accepts a base64 image (`image`) and saves it to `static/captures`

## Security & Production notes

- `app.secret_key` in `webapp/app.py` is a placeholder; change it before deploying.
- Running TensorFlow and webcam access in production requires careful resource handling; consider running model inference in a separate worker or using batching for scalability.

## Troubleshooting

- Pylance warnings about missing packages typically mean your VS Code interpreter isn't the one with your installed packages — select the correct Python interpreter in the bottom-right of VS Code.
- If the camera fails to open, ensure no other app is using it and try changing the device index.

## Credits

This repository stitches together OpenCV face detection and a pretrained Keras emotion model to demonstrate real-time emotion recognition in the browser.

---
If you want, I can also create `requirements.txt`, add a short `README` badge header, or help containerize this with Docker.
