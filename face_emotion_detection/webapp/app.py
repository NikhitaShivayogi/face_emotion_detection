from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from flask_socketio import SocketIO, emit
import os
import io, base64
from typing import Dict
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load emotion model
model_path = os.path.join(os.path.dirname(__file__), '../src/emotion_model.h5')
emotion_model = load_model(model_path, compile=False)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face cascade
cascade_path = os.path.join(os.path.dirname(__file__), '../src/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Store latest emotion data
latest_emotions: Dict[str, float] = {label: 0.0 for label in emotion_labels}

def generate_frames():
    global latest_emotions
    global cap
    while True:
        # ensure capture is open (re-open if previously released)
        try:
            if cap is None or (hasattr(cap, 'isOpened') and not cap.isOpened()):
                cap = cv2.VideoCapture(0)
        except Exception:
            cap = cv2.VideoCapture(0)

        success, frame = cap.read()
        if not success:
            # if reading fails, wait briefly and retry (prevents tight loop)
            import time
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (64, 64))
            roi_gray = roi_gray / 255.0
            roi_gray = np.reshape(roi_gray, (1, 64, 64, 1))

            preds = emotion_model.predict(roi_gray, verbose=0)
            emotion_idx = np.argmax(preds)
            emotion = emotion_labels[emotion_idx]
            
            # Update latest emotions with percentages
            for i, label in enumerate(emotion_labels):
                latest_emotions[label] = float(preds[0][i] * 100)
            
            # Emit emotion data via WebSocket
            socketio.emit('emotion_update', latest_emotions)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 107, 157), 3)
            cv2.putText(frame, f"{emotion} {preds[0][emotion_idx]*100:.1f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 107, 157), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    # Open access dashboard — no login required
    username = session.get('user', 'Guest')
    return render_template('dashboard.html', username=username)


@app.route('/index')
def index():
    # Detection UI (main demo page)
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotions')
def get_emotions():
    return jsonify(latest_emotions)

@app.route('/capture', methods=['POST'])
def capture():
    try:
        data = request.get_json()
        img_data = data.get('image', '')
        if not img_data.startswith('data:image'):
            return jsonify({'error': 'invalid image data'}), 400

        header, encoded = img_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)

        save_dir = os.path.join(os.path.dirname(__file__), 'static', 'captures')
        os.makedirs(save_dir, exist_ok=True)
        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(save_dir, filename)

        with open(path, 'wb') as f:
            f.write(img_bytes)

        return jsonify({'filename': f'static/captures/{filename}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Release the global camera capture so the server frees the webcam."""
    global cap
    try:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
        return jsonify({'status': 'stopped'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    socketio.run(app, debug=True)
