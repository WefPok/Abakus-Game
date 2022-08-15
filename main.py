import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.python.keras.models import load_model

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')


# center_relX - относительное положение зоны по x, Например 0.5 значит что центр зоны будет посередине картинки по
# горизонтали

# center_relY - относительное положение зоны по x, Например 0.5 значит что центр зоны будет посередине
# картинки по вертикали

# width, height ширина и высота зоны в пикселях
# frame - кадр с потока, чтобы определить высоту и ширину входящей картинки
# isupper True если это небесная косточка
# cost - стоимость косточки
class Zone:
    def __init__(self, center_relX, center_relY, width, height, frame, isupper, cost):
        self.width = width
        self.height = height
        self.center_X = int(center_relX * frame.shape[1])
        self.center_Y = int(center_relY * frame.shape[0])
        self.isupper = isupper
        self.cost = cost

    active = False
    activator = None
    color = (255, 0, 0)

    def activate(self, finger):
        self.color = (0, 0, 255)
        self.activator = finger

    def deactivate(self):
        self.color = (255, 0, 0)
        self.activator = None

    def isactive(self, finger):
        if finger is None:
            return False
        if self.center_X - int(self.width / 2) < finger.x < self.center_X + int(self.width / 2) and self.center_Y - int(
                self.height / 2) < finger.y < self.center_Y + int(self.height / 2):
            return True
        else:
            return False

    def exit(self, finger):
        if self.center_Y - int(self.height / 2) > finger.y and self.center_X - int(
                self.width / 2) < finger.x < self.center_X + int(
                self.width / 2):
            return "up"
        elif self.center_Y + int(self.height / 2) < finger.y and self.center_X - int(
                self.width / 2) < finger.x < self.center_X + int(
                self.width / 2):
            return "down"
        else:
            return "neutral"


class Finger:
    def __init__(self, id):
        self.x = 0
        self.y = 0
        self.id = id


zones = []


def draw_grid(frame, zones):
    for zone in zones:
        cv2.rectangle(frame, (zone.center_X - int(zone.width / 2), zone.center_Y - int(zone.height / 2)),
                      (zone.center_X + int(zone.width / 2), zone.center_Y + int(zone.height / 2)), zone.color, 2)

    return frame


cap = cv2.VideoCapture(0)
_, frame = cap.read()

zones.append(Zone(0.25, 0.4, 150, 40, frame, True, 500))
zones.append(Zone(0.5, 0.4, 150, 40, frame, True, 50))
zones.append(Zone(0.75, 0.4, 150, 40, frame, True, 5))
zones.append(Zone(0.25, 0.55, 150, 40, frame, False, 100))
zones.append(Zone(0.5, 0.55, 150, 40, frame, False, 10))
zones.append(Zone(0.75, 0.55, 150, 40, frame, False, 1))

index_right = Finger("ir")
index_left = Finger("il")

thumb_right = Finger("tr")
thumb_left = Finger("tl")
fingers = [thumb_right, thumb_left, index_right, index_left]

total_sum = 0

while True:
    _, frame = cap.read()
    y, x, c = frame.shape
    # Flip the frame vertically
    # frame = cv2.flip(frame, 1)
    # Show the final output

    # release the webcam and destroy all active windows
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hand landmark prediction
    result = hands.process(framergb)
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        index_fingers = []
        thumb_fingers = []
        if len(result.multi_hand_landmarks) < 2:
            print("Can't see both hands")
            continue

        hand1 = result.multi_hand_landmarks[0]
        hand2 = result.multi_hand_landmarks[1]

        index_right.x = int(hand1.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * x)
        index_right.y = int(hand1.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * y)

        thumb_right.x = int(hand1.landmark[mpHands.HandLandmark.THUMB_TIP].x * x)
        thumb_right.y = int(hand1.landmark[mpHands.HandLandmark.THUMB_TIP].y * y)

        index_left.x = int(hand2.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * x)
        index_left.y = int(hand2.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * y)

        thumb_left.x = int(hand2.landmark[mpHands.HandLandmark.THUMB_TIP].x * x)
        thumb_left.y = int(hand2.landmark[mpHands.HandLandmark.THUMB_TIP].y * y)

        for finger in fingers:
            if finger.id[1] == "l":
                cv2.circle(frame, (finger.x, finger.y), 5, (0, 255, 0), -1)
            else:
                cv2.circle(frame, (finger.x, finger.y), 5, (0, 0, 255), -1)

        for zone in zones:
            for finger in fingers:
                if zone.isactive(finger):
                    zone.activate(finger)

            if zone.activator is None:
                continue

            if not zone.isactive(zone.activator):
                res = zone.exit(zone.activator)
                print(res, zone.activator.id)

                if res == "down" and zone.activator.id[0] == 'i':
                    total_sum -= zone.cost
                if res == "up" and zone.activator.id[0] == 'i' and zone.isupper:
                    total_sum += zone.cost
                if res == "up" and zone.activator.id[0] == 't' and not zone.isupper:
                    total_sum += zone.cost

                zone.deactivate()

        for zone in zones:
            if zone.activator is not None:
                cv2.putText(frame, zone.activator.id, (zone.center_X, zone.center_Y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            zone.color, 2)

    cv2.putText(frame, str(total_sum), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    draw_grid(frame, zones)
    frame = cv2.flip(frame, 1)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
