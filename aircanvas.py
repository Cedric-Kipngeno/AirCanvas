import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # change 0 to 1 if the window is black

colors = {"1": (0, 0, 255), "2": (0, 255, 0), "3": (255, 0, 0)}  # BGR
color = colors["1"]
brush = 6
canvas = None
prev = None       # last fingertip position
smooth = None     # smoothed fingertip position
timestamp = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read from the camera")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp += 33
    result = detector.detect_for_video(image, timestamp)

    mode = "No hand"
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # fingertip position, smoothed so lines are less shaky
        x, y = int(hand[8].x * w), int(hand[8].y * h)
        if smooth is None:
            smooth = (x, y)
        else:
            smooth = (int(smooth[0] * 0.6 + x * 0.4),
                      int(smooth[1] * 0.6 + y * 0.4))

        index_up = hand[8].y < hand[6].y
        middle_up = hand[12].y < hand[10].y

        if index_up and not middle_up:
            mode = "Drawing"
            if prev is not None:
                cv2.line(canvas, prev, smooth, color, brush)
            prev = smooth
        else:
            mode = "Pen up"
            prev = None

        cv2.circle(frame, smooth, 10, color, -1)
    else:
        prev = None
        smooth = None

    # put the drawing on top of the live video
    mask = canvas.any(axis=2)
    frame[mask] = canvas[mask]

    cv2.putText(frame, mode, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Air Canvas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas[:] = 0
    elif key == ord("s"):
        cv2.imwrite("drawing.png", canvas)
        print("Saved drawing.png")
    elif chr(key) in colors:
        color = colors[chr(key)]

cap.release()
cv2.destroyAllWindows()