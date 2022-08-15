import cv2
from tensorflow.python.keras.models import load_model
import mediapipe as mp
import json

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8)

model = load_model('mp_hand_gesture')


class Finger:
    def __init__(self, id):
        self.x = 0
        self.y = 0
        self.id = id


index_right = Finger("ir")
index_left = Finger("il")

thumb_right = Finger("tr")
thumb_left = Finger("tl")
fingers = [thumb_right, thumb_left, index_right, index_left]


def create_json(hand_lanmark):

    accepted_points = [0, 4, 8, 12, 16, 20]
    names = ['wrist', 'thumb', 'index', 'middle', 'ring', 'pinky']
    res = {}
    for index, point in enumerate(hand_lanmark):
        if index in accepted_points:
            temp = {'x': round(point.x, 2),
                    'y': round(point.y, 2),
                    'z': round(point.z, 2)}
            res[names[accepted_points.index(index)]] = temp
    return json.dumps(res)

def recognize(cap):
    _, frame = cap.read()
    y, x, c = frame.shape
    # Flip the frame vertically
    # frame = cv2.flip(frame, 1)
    # Show the final output

    # release the webcam and destroy all active windows
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hand landmark prediction
    result = hands.process(framergb)
    # post process the result

    if result.multi_hand_world_landmarks:

        if len(result.multi_hand_world_landmarks) < 2:
            print("Can't see both hands")
            return "No hand"

        hand1 = result.multi_hand_world_landmarks[0]
        hand2 = result.multi_hand_world_landmarks[1]

        json = create_json(hand1.landmark)


        index_right.x = hand1.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x
        index_right.y = hand1.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y

        thumb_right.x = hand1.landmark[mpHands.HandLandmark.THUMB_TIP].x
        thumb_right.y = hand1.landmark[mpHands.HandLandmark.THUMB_TIP].y

        index_left.x = hand2.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x
        index_left.y = hand2.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y

        thumb_left.x = hand2.landmark[mpHands.HandLandmark.THUMB_TIP].x
        thumb_left.y = hand2.landmark[mpHands.HandLandmark.THUMB_TIP].y

        return json
    else:
        return "No both hands"
