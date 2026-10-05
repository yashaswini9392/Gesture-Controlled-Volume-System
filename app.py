import cv2
import math
import numpy as np
import mediapipe as mp
from pycaw.pycaw import AudioUtilities

# 1. Initialize Audio
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume
min_vol, max_vol, _ = volume.GetVolumeRange()

# 2. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 3. Webcam Feed
cap = cv2.VideoCapture(0)

vol_bar = 400
vol_per = 0

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

            if landmarks:
                x1, y1 = landmarks[4]   # Thumb tip
                x2, y2 = landmarks[8]   # Index tip
                
                wrist_x, wrist_y = landmarks[0]
                fingertip_ids = [8, 12, 16, 20]
                
                extended_fingers = 0
                for tip_id in fingertip_ids:
                    dist_to_wrist = math.hypot(landmarks[tip_id][0] - wrist_x, landmarks[tip_id][1] - wrist_y)
                    if dist_to_wrist > 100:
                        extended_fingers += 1

                # Fist = Mute, Extended Hand = Volume Control
                if extended_fingers == 0:
                    volume.SetMute(1, None)
                    cv2.putText(img, "MUTED", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    volume.SetMute(0, None)

                    distance = math.hypot(x2 - x1, y2 - y1)

                    vol = np.interp(distance, [20, 200], [min_vol, max_vol])
                    vol_bar = np.interp(distance, [20, 200], [400, 150])
                    vol_per = np.interp(distance, [20, 200], [0, 100])

                    volume.SetMasterVolumeLevel(vol, None)

                    cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if distance < 25:
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    # Visual Volume Bar UI
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()