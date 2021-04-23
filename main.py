import math
import os
import sys
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


def findLineEquation(points):
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    a, b = np.linalg.lstsq(A, y_coords)[0]
    return {"a": a, "b": b}


class Piro:
    def __init__(self, image):
        self.image = image
        self.basePoints = tuple()
        self.armPoints1 = tuple()
        self.armPoints2 = tuple()
        self.armFunction1 = dict()
        self.armFunction2 = dict()

    def solve(self):
        contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgCopy = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2BGR)

        epsilon = 0.005 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        print(len(approx))
        points = []
        allPoints = []

        for i in range(-2, len(approx) - 2):
            allPoints.append((approx[i][0][0], approx[i][0][1]))
            points.append(((approx[i][0][0], approx[i][0][1]), (approx[i + 1][0][0], approx[i + 1][0][1]),
                           (approx[i + 2][0][0], approx[i + 2][0][1])))

        angles = []
        for p in points:
            angles.append(angle(p))
        shiftedAngles = shiftAngleArray(angles)
        print(shiftedAngles)
        self.basePoints = shiftedAngles[0][1], shiftedAngles[0][2]
        self.armPoints1 = (shiftedAngles[0][0], shiftedAngles[0][1])
        self.armPoints2 = (shiftedAngles[1][1], shiftedAngles[1][2])

        self.armFunction1 = findLineEquation(self.armPoints1)
        self.armFunction2 = findLineEquation(self.armPoints2)

        # print("functions", self.armPoints1, self.armFunction1)

        cv2.drawContours(imgCopy, [approx], 0, color=(0, 0, 255), thickness=2)
        cv2.imshow("image", imgCopy)
        cv2.waitKey()


if __name__ == '__main__':
    images = dict()
    for n in range(int(sys.argv[2])):
        images[n] = cv2.imread(os.path.join(os.path.dirname(sys.argv[1]), str(n) + ".png"), cv2.IMREAD_GRAYSCALE)

    for key, im in images.items():
        piroObject = Piro(im)
        piroObject.solve()
