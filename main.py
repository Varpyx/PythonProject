import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # levá ruka
    (12, 14), (14, 16),  # pravá ruka
    (11, 12),            # ramena

    (11, 23), (12, 24),  # trup
    (23, 24),            # boky

    (23, 25), (25, 27),  # levá noha
    (24, 26), (26, 28),  # pravá noha

    (27, 31), (28, 32)   # chodidla
]

model_path = "pose_landmarker_full.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)

cap = cv2.VideoCapture("squat.mp4")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # Převod na pixely
            points = []
            for lm in landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                points.append((x, y))

            # 🔵 vykreslení bodů
            for (x, y) in points:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # 🔴 vykreslení spojení
            for a, b in POSE_CONNECTIONS:
                cv2.line(frame, points[a], points[b], (255, 0, 0), 2)

        cv2.imshow("Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()