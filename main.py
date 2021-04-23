import math
import os
import random
import sys
from typing import List

import cv2
import numpy as np


def angle(points):
    ang = math.degrees(math.atan2(points[2][1] - points[1][1], points[2][0] - points[1][0]) - math.atan2(
        points[0][1] - points[1][1], points[0][0] - points[1][0]))

    if ang < 0:
        return points[0], points[1], points[2], ang + 360
    return points[0], points[1], points[2], ang


def shiftAngleArray(angles):
    minimum = 360
    minDiffSum = 360
    for i in range(-1, len(angles) - 1):
        diff = 90 - angles[i][3]
        if diff < 0:
            diff = 0 - diff
        if diff < minimum:
            if 85 < angles[i + 1][3] < 95:
                diff2 = 90 - angles[i + 1][3]
                if diff2 < 0:
                    diff2 = 0 - diff2
                diffSum = diff + diff2
                if diffSum < minDiffSum:
                    possibleBaseAngles = i
                    minDiffSum = diffSum
    return angles[possibleBaseAngles:] + angles[:possibleBaseAngles]


def calculate_peaks_distance(points, base):
    p1, p2 = base

    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = -1 * (a * (p2[0]) + b * (p2[1]))
    distances = []

    base_length = math.hypot(p1[1] - p1[0], p2[1] - p2[0])

    for p3 in points:
        x, y = p3[1][0], p3[1][1]
        distance = abs(a * x + b * y + c) / math.sqrt(a ** 2 + b ** 2)
        distances.append(distance / base_length)
    print(distances)
    return distances


def evaluate(correct_list: List, results):
    scores = []
    for i in range(0, len(results)):
        score = 0
        for j in range(0, len(results[i])):
            if correct_list[i] == results[i][j]:
                score = 1 / (j + 1)
                break
        if score == 0:
            score = 1 / len(correct_list)
        scores.append(score)
    return sum(scores) / len(scores)


class Piro:
    def __init__(self):
        self.images = dict()

    def load(self, directory, numberOfImages):
        for n in range(numberOfImages):
            print(directory)
            self.images[n] = cv2.imread(os.path.join(directory, str(n) + ".png"), cv2.IMREAD_GRAYSCALE)

    def solve(self):
        for key, im in self.images.items():
            contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            imgCopy = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2BGR)

            epsilon = 0.005 * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True)
            print(len(approx))
            points = []

            for i in range(-2, len(approx) - 2):
                points.append(((approx[i][0][0], approx[i][0][1]), (approx[i + 1][0][0], approx[i + 1][0][1]),
                               (approx[i + 2][0][0], approx[i + 2][0][1])))

            angles = []
            for p in points:
                angles.append(angle(p))
            shiftedAngles = shiftAngleArray(angles)
            print(shiftedAngles)
            base_points = shiftedAngles[0][1], shiftedAngles[0][2]
            print(imgCopy.shape)
            calculate_peaks_distance(shiftedAngles[2:], base_points)
            cv2.line(imgCopy, base_points[0], base_points[1], (0, 255, 0))

            cv2.drawContours(imgCopy, [approx], 0, color=(0, 0, 255), thickness=2)
            cv2.imshow("image", imgCopy)
            cv2.waitKey()

        return [[0]] * len(self.images.keys())


if __name__ == '__main__':
    piroObject = Piro()
    piroObject.load(os.path.dirname(sys.argv[1]), int(sys.argv[2]))
    results = piroObject.solve()  # results should be a list with best matches, np. [[1,2],[4,1]]
    print(results)
    with open(sys.argv[1] + 'correct.txt', 'r') as f:
        correct_list = []
        for line in f.readlines():
            correct_list.append(int(line))
        print(correct_list)
        print(evaluate(correct_list, results))
