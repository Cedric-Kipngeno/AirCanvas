import cv2
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
timestamp = 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read from the camera")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp += 33  # milliseconds, must keep increasing
    result = detector.detect_for_video(image, timestamp)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        for point in hand:  # draw all 21 landmarks
            cv2.circle(frame, (int(point.x * w), int(point.y * h)), 4, (255, 255, 0), -1)

        tip = hand[8]  # index fingertip
        x, y = int(tip.x * w), int(tip.y * h)
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        print("Index fingertip:", x, y)

    cv2.imshow("Hand test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()