import math
import time

import cv2
from tensorflow.python.keras.models import load_model
import mediapipe as mp
import json

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.8)

model = load_model('mp_hand_gesture')


class EventStack:
    def __init__(self, length, placeholder=2):
        self.stack = [placeholder] * length
        self.length = length
        self.placeholder = placeholder

    def append(self, event):
        self.stack.pop(0)
        self.stack.append(event)

    def get_last(self):
        return self.stack[-1]

    def get_stack(self):
        return self.stack

    def clear(self):
        self.stack = [self.placeholder] * self.length


def get_x_range(wrist_x):
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


# def create_json(hand_lanmark1, hand_lanmark2, analyze_res1, analyze_res2):
#     res = {}
#     wrist_x = hand_lanmark1[0].x
#     wrist_x = get_x_range(wrist_x)
#
#     res['right_hand'] = {"x_position": wrist_x,
#                          "events": analyze_res1}
#
#     wrist_x = hand_lanmark2[0].x
#     wrist_x = get_x_range(wrist_x)
#
#     res['left_hand'] = {"x_position": wrist_x,
#                         "events": analyze_res2}
#
#     return json.dumps(res)

class Hand:
    def __init__(self, hand_id):
        self.hand_id = hand_id
        if self.hand_id == 0:
            self.name = "left"
        else:
            self.name = "right"
        self.thumb_plain_stack = EventStack(20, 0)
        self.index_plain_stack = EventStack(20, 0)

        self.poi = None

        self.path_total_index = EventStack(10, 0)
        self.angle_index = EventStack(10, 0)

        self.path_total_thumb = EventStack(10, 0)
        self.angle_thumb = EventStack(10, 0)
        self.x_pos = 0

        self.index_bent = False
        self.index_moved = 0
        self.index_timeout = 0

        self.thumb_bent = False
        self.thumb_timeout = 0
        # self.thumb_moved = False

    def update_hand_position(self, points):
        self.x_pos = get_x_range(points[-1].x)
        y_pos = points[-1].y

        self.thumb_plain_stack.append(points[0].y - y_pos)
        self.index_plain_stack.append(points[3].y - y_pos)

        self.poi = points

        index_plain = self.index_plain_stack.get_stack()
        self.path_total_index.append(int(1000 * sum(
            [abs(i - j) for i, j in zip(index_plain[1:], index_plain[2:-1])])))

        thumb_plain = self.thumb_plain_stack.get_stack()
        self.path_total_thumb.append(int(1000 * sum(
            [abs(i - j) for i, j in zip(thumb_plain[1:], thumb_plain[2:-1])])))

        ip2, ip1, ip3 = self.poi[3:6]
        self.angle_index.append(abs(
            180 - abs(int(math.degrees(
                math.atan2(ip3.y - ip1.y, ip3.x - ip1.x) - math.atan2(ip2.y - ip1.y, ip2.x - ip1.x))))))

        tp2, tp1, tp3 = self.poi[0:3]
        self.angle_thumb.append(abs(
            180 - abs(int(math.degrees(
                math.atan2(tp3.y - tp1.y, tp3.x - tp1.x) - math.atan2(tp2.y - tp1.y, tp2.x - tp1.x))))))

    def set_state_variables(self):
        events = {"+1": False,
                  "-1": False,
                  "+5": False,
                  "-5": False}
        if self.thumb_timeout == 0:
            thumb_first5 = self.angle_thumb.get_stack()[0:5]
            if all(i < 40 for i in thumb_first5):
                if self.thumb_bent:
                    self.thumb_bent = False
            else:
                if not self.thumb_bent:
                    print("+1 Thumb " + self.name)
                    events["+1"] = True
                    self.thumb_bent = True
                    self.thumb_timeout = 10
                    self.angle_thumb.clear()
        else:
            self.thumb_timeout -= 1

        if self.index_timeout == 0:
            index_first5 = self.angle_index.get_stack()[0:5]
            if all(i < 40 for i in index_first5):
                if self.index_bent:
                    self.index_bent = False
                if sum(self.path_total_index.get_stack()) > 1200:
                    if not self.index_moved:
                        print("-1 Index " + self.name)
                        events["-1"] = True
                        self.index_moved = True
                        self.index_timeout = 10
                elif self.index_moved:
                    self.index_moved = False
            else:
                if not self.index_bent and not self.index_moved:
                    self.index_bent = True
                    self.index_moved = True
                    self.index_timeout = 10
                    if sum(self.path_total_index.get_stack()) > 700:
                        events["-5"] = True
                        print("-5 Index " + self.name)
                    else:
                        events["+5"] = True
                        print("+5 Index " + self.name)
        else:
            self.index_timeout -= 1
        return events


class Analyzer:
    def __init__(self):
        self.hands = [Hand(0), Hand(1)]

    def update_hands(self, frame, show=True):
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)

        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
            hand1 = result.multi_hand_landmarks[0].landmark
            hand2 = result.multi_hand_landmarks[1].landmark

            points1 = [hand1[4], hand1[3], hand1[1], hand1[8], hand1[6], hand1[5], hand1[1]]
            points2 = [hand2[4], hand2[3], hand2[1], hand2[8], hand2[6], hand2[5], hand2[1]]

            if hand1[0].x < hand2[0].x:
                self.hands[0].update_hand_position(points1)
                self.hands[1].update_hand_position(points2)
            else:
                self.hands[0].update_hand_position(points2)
                self.hands[1].update_hand_position(points1)
            if show:
                h, w = frame.shape[:2]
                copy = frame.copy()
                for point in self.hands[0].poi:
                    x = int(point.x * w)
                    y = int(point.y * h)
                    copy = cv2.circle(copy, (x, y), 2, (0, 255, 0), -1)
                for point in self.hands[1].poi:
                    x = int(point.x * w)
                    y = int(point.y * h)
                    copy = cv2.circle(copy, (x, y), 2, (255, 0, 0), -1)

                cv2.imshow("handlandmarks", copy)

    def main(self, frame):

        self.update_hands(frame)
        # print(self.hands[0].angle_index.get_stack())
        event_json = {"right_hand": {"x_position": self.hands[0].x_pos,
                                     "events": self.hands[0].set_state_variables()},

                      "left_hand": {"x_position": self.hands[1].x_pos,
                                    "events": self.hands[1].set_state_variables()}}

        return json.dumps(event_json)
