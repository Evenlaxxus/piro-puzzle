import math
import os
import sys
import cv2 as cv
import numpy as np


def angle(points):
    ang = math.degrees(math.atan2(points[2][1] - points[1][1], points[2][0] - points[1][0]) - math.atan2(
        points[0][1] - points[1][1], points[0][0] - points[1][0]))
    return ang + 360 if ang < 0 else ang


def shiftAngleArray(angles):
    minimum = 360
    minDiffSum = 360
    for i in range(-1, len(angles) - 1):
        diff = 90 - angles[i]
        if diff < 0:
            diff = 0 - diff
        if diff < minimum:
            diff2 = 90 - angles[i + 1]
            if 85 < angles[i + 1] < 95:
                diff2 = 90 - angles[i + 1]
                if diff2 < 0:
                    diff2 = 0 - diff2
                diffSum = diff + diff2
                if diffSum < minDiffSum:
                    possibleBaseAngles = i
                    minDiffSum = diffSum
    return angles[possibleBaseAngles:] + angles[:possibleBaseAngles]


class Piro:
    def __init__(self):
        self.images = dict()

    def load(self, directory, numberOfImages):
        for n in range(numberOfImages):
            self.images[n] = cv.imread(os.path.join(directory, str(n) + ".png"), cv.IMREAD_GRAYSCALE)

    def solve(self):
        for key, im in self.images.items():
            contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            imgCopy = cv.cvtColor(im.copy(), cv.COLOR_GRAY2BGR)

            epsilon = 0.0069 * cv.arcLength(contours[0], True)
            approx = cv.approxPolyDP(contours[0], epsilon, True)
            print(len(approx))
            points = []

            for i in range(-2, len(approx) - 2):
                points.append(((approx[i][0][0], approx[i][0][1]), (approx[i + 1][0][0], approx[i + 1][0][1]),
                               (approx[i + 2][0][0], approx[i + 2][0][1])))

            angles = []
            for p in points:
                angles.append(angle(p))
            print(shiftAngleArray(angles))

            cv.drawContours(imgCopy, [approx], 0, color=(0, 0, 255), thickness=2)
            cv.imshow("image", imgCopy)
            cv.waitKey()


if __name__ == '__main__':
    piroObject = Piro()
    piroObject.load(os.path.dirname(sys.argv[1]), int(sys.argv[2]))
    piroObject.solve()
