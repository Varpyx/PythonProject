import math
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python


# Funkce pro výpočet úhlu ze tří bodů
def calculate_angle(a, b, c):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


# Funkce pro výpočet vzdálenosti dvou bodů (Pythagorova věta)
def calculate_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # levá ruka
    (12, 14), (14, 16),  # pravá ruka
    (11, 12),  # ramena
    (11, 23), (12, 24),  # trup
    (23, 24),  # boky
    (23, 25), (25, 27),  # levá noha
    (24, 26), (26, 28),  # pravá noha
    (27, 31), (28, 32),
    (30, 32), (29, 31),  # chodidla
    (28, 30), (27, 29)
]

model_path = "pose_landmarker_full.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False
)

cap = cv2.VideoCapture("squat_front.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = 0
counter = 0
stage = "up"

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_index += 1
        timestamp_ms = int(frame_index * (1000 / fps))

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            points = []
            for lm in landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                points.append((x, y))

            # 🔵 Vykreslení spojení a bodů (standardně modře a zeleně)
            for a, b in POSE_CONNECTIONS:
                cv2.line(frame, points[a], points[b], (255, 0, 0), 2)
            for (x, y) in points:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            hip = points[24]
            knee = points[26]
            ankle = points[28]
            angle = calculate_angle(hip, knee, ankle)

            # --- BIOMECHANIKA ZEPŘEDU ---
            # Získání bodů pro obě nohy
            left_knee = points[25]
            right_knee = points[26]
            left_ankle = points[27]
            right_ankle = points[28]

            # Výpočet vzdáleností
            knee_dist = calculate_distance(left_knee, right_knee)
            ankle_dist = calculate_distance(left_ankle, right_ankle)

            # Detekce propadání kolen (pokud je vzdálenost kolen menší než 75% vzdálenosti kotníků)
            # Číslo 0.75 můžeš upravit podle toho, jak přísný má program být
            is_actively_squatting = angle < 160
            potentially_valgus = knee_dist < (ankle_dist * 0.70)
            is_valgus_error = potentially_valgus and is_actively_squatting

            if is_valgus_error:
                # Zvýrazníme chybu nakreslením tlusté červené čáry mezi koleny
                cv2.line(frame, left_knee, right_knee, (0, 0, 255), 4)
                cv2.putText(frame, "CHYBA: KOLENA K SOBE!", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # --- POČÍTÁNÍ DŘEPŮ (Ponecháno z původního kódu) ---


            if angle > 170:
                stage = "up"
            if angle < 90 and stage == "up":
                stage = "down"
            if angle > 160 and stage == "down":
                stage = "up"
                counter += 1

            # --- VYKRESLENÍ UI ---
            cv2.rectangle(frame, (0, 0), (450, 150), (0, 0, 0), -1)
            cv2.putText(frame, f'Opakovani: {counter}', (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Faze: {stage}', (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            angle_color = (0, 255, 0) if angle <= 90 else (255, 255, 255)
            cv2.putText(frame, f'Uhel (prava): {int(angle)}', (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, angle_color, 2)

        cv2.imshow("Biomechanika Drepu", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()