import cv2
import os

# Path to Haar Cascade file
cascade_path = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

    cv2.imshow("Face Detection - Press Q to Exit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
