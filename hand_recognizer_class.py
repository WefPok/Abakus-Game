import cv2
from tensorflow.python.keras.models import load_model
import mediapipe as mp
import json

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8)

model = load_model('mp_hand_gesture')


class EventStack:
    def __init__(self, length, placeholder="PlaceHolder"):
        self.stack = [placeholder] * length

    def append(self, event):
        self.stack.pop(0)
        self.stack.append(event)

    def get_last(self):
        return self.stack[-1]

    def get_stack(self):
        return self.stack


def event_analyzer(position_stack: EventStack, hand_landmark):
    wrist_position = hand_landmark[0].y
    thumb_position = hand_landmark[4].y - wrist_position
    index_position = hand_landmark[8].y - wrist_position

    position_stack.append([thumb_position, index_position])


def get_x_range(wrist_x):
    res = 9
    if wrist_x < 0.1:
        res = 0
    elif 0.1 <= wrist_x < 0.2:
        res = 1
    elif 0.2 <= wrist_x < 0.3:
        res = 2
    elif 0.3 <= wrist_x < 0.4:
        res = 3
    elif 0.4 <= wrist_x < 0.5:
        res = 4
    elif 0.5 <= wrist_x < 0.6:
        res = 5
    elif 0.6 <= wrist_x < 0.7:
        res = 6
    elif 0.7 <= wrist_x < 0.8:
        res = 7
    elif 0.8 <= wrist_x < 0.9:
        res = 8
    else:
        res = 9
    return res


def create_json(hand_lanmark1, hand_lanmark2):
    accepted_points = [0, 4, 8, 12, 16, 20, 9]
    names = ['wrist', 'thumb', 'index', 'middle', 'ring', 'pinky', 'directional']
    res = {}
    wrist_x = hand_lanmark1[4].x
    wrist_x = get_x_range(wrist_x)

    res['right_hand'] = {"x_position": wrist_x}

    wrist_x = hand_lanmark2[4].x
    wrist_x = get_x_range(wrist_x)

    res['left_hand'] = {"x_position": wrist_x}



    # for index, point in enumerate(hand_lanmark):
    #     if index in accepted_points:
    #         temp = {'x': round(point.x, 2),
    #                 'y': round(point.y, 2),
    #                 'z': round(point.z, 2)}
    #         res[names[accepted_points.index(index)]] = temp
    return json.dumps(res)


def recognize(cap):
    positions_stack = EventStack(30)
    _, frame = cap.read()
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:

        hand1 = result.multi_hand_landmarks[0]
        hand2 = result.multi_hand_landmarks[1]
        json = create_json(hand1.landmark, hand2.landmark)

        return json
    else:
        return None
