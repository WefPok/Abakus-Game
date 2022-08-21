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
        self.length = length

    def append(self, event):
        self.stack.pop(0)
        self.stack.append(event)

    def get_last(self):
        return self.stack[-1]

    def get_stack(self):
        return self.stack

    def clear(self):
        self.stack = ["PlaceHolder"] * self.length


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


def create_json(hand_lanmark1, hand_lanmark2, analyze_res1, analyze_res2):
    res = {}
    wrist_x = hand_lanmark1[4].x
    wrist_x = get_x_range(wrist_x)

    res['right_hand'] = {"x_position": wrist_x,
                         "events": analyze_res1}

    wrist_x = hand_lanmark2[4].x
    wrist_x = get_x_range(wrist_x)

    res['left_hand'] = {"x_position": wrist_x,
                        "events": analyze_res2}

    return json.dumps(res)


class Recognizer:
    def __init__(self):
        self.thumb_stack1 = EventStack(10)
        self.thumb_stack2 = EventStack(10)
        self.thumb_state1 = True
        self.thumb_state2 = True

        self.index_stack1 = EventStack(10)
        self.index_stack2 = EventStack(10)
        self.index_state1 = True
        self.index_state2 = True

    def recognize(self, frame):
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)

        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:

            hand1 = result.multi_hand_landmarks[0]
            hand2 = result.multi_hand_landmarks[1]

            analysis1 = self.event_analyzer(self.thumb_stack1, self.index_stack1, hand1.landmark, self.thumb_state1,
                                            self.index_state1, 1)
            analysis2 = self.event_analyzer(self.thumb_stack2, self.index_stack2, hand2.landmark, self.thumb_state2,
                                            self.index_state2, 2)

            json = create_json(hand1.landmark, hand2.landmark, analysis1, analysis2)

            return json
        else:
            return None

    def event_analyzer(self, thumb_pos: EventStack, index_pos: EventStack, hand_landmark, thumb_state, index_state,
                       hand_n):
        wrist_position = hand_landmark[0].y
        thumb_position = hand_landmark[4].y - wrist_position
        index_position = hand_landmark[8].y - wrist_position

        thumb_pos.append(thumb_position)
        index_pos.append(index_position)

        res = {"+5": False,
               "+1": False,
               "-5": False,
               "-1": False}

        if "PlaceHolder" not in thumb_pos.get_stack():
            diff_thumb = thumb_position - thumb_pos.get_stack()[-5]

            if abs(diff_thumb) > 0.1 and diff_thumb > 0 and thumb_state:
                res["+1"] = True
                if hand_n == 1:
                    self.thumb_state1 = False
                    self.thumb_stack1.clear()
                else:
                    self.thumb_state2 = False
                    self.thumb_stack2.clear()
            elif not thumb_state:
                if hand_n == 1:
                    self.thumb_state1 = True
                else:
                    self.thumb_state2 = True

        if "PlaceHolder" not in index_pos.get_stack():
            diff_index = index_position - index_pos.get_stack()[-5]
            if 0.1 < abs(diff_index) and diff_index > 0 and index_state:
                res["+5"] = True
                if hand_n == 1:
                    self.index_state1 = False
                    self.index_stack1.clear()
                else:
                    self.index_state2 = False
                    self.index_stack2.clear()

            elif 0.1 < abs(diff_index) < 0.2 and diff_index < 0 and index_state:
                res["-5"] = True
                if hand_n == 1:
                    self.index_state1 = False
                    self.index_stack1.clear()
                else:
                    self.index_state2 = False
                    self.index_stack2.clear()

            elif abs(diff_index) > 0.2 and diff_index < 0 and index_state:
                res["-1"] = True
                if hand_n == 1:
                    self.index_state1 = False
                    self.index_stack1.clear()
                else:
                    self.index_state2 = False
                    self.index_stack2.clear()

            elif not thumb_state:
                if hand_n == 1:
                    self.thumb_state1 = True
                else:
                    self.thumb_state2 = True
        return res
