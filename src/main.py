import math
import os
import sys
from pathlib import Path
from statistics import stdev
from typing import List
import numpy as np
import cv2


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
    def __init__(self, image, name):
        self.name = name
        self.image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT)
        self.basePoints = tuple()
        self.baseLength = 1
        self.allPoints = []
        self.distances = []
        self.pointsToCheck = []
        self.contours = None
        self.hierarchy = None
        self.armsPoints = tuple()
        self.curvePoints = []
        self.contour = []

    def process(self):
        _ = self.getMetrics()
        self.image = self.rotateImage()
        approx, imgCopy = self.getMetrics()
        if self.basePoints[0][1] < self.image.shape[:2][0] / 2:
            self.image = self.rotateImage(flip="yes")
            approx, imgCopy = self.getMetrics()

        self.getEdgePoints()
        self.getCurvePointsToCheck()
        self.getDistances()

        cv2.drawContours(imgCopy, [approx], 0, color=(0, 0, 255), thickness=2)
        for point in self.pointsToCheck:
            cv2.circle(imgCopy, point, radius=0, color=(0, 255, 0), thickness=6)

        # print(self.basePoints)
        # cv2.imshow("image", imgCopy)
        # cv2.waitKey()

    def getBaseAndShift(self, cont):
        maxLength = 0.0
        points = []
        bPoints = []
        for i in range(-1, len(cont) - 1):
            length = math.dist(cont[i][0], cont[i + 1][0])
            if length > maxLength:
                bPoints = cont[i][0], cont[i + 1][0]
                maxLength = length
            points.append(tuple(cont[i][0]))

        self.basePoints = (tuple(bPoints[0]), tuple(bPoints[1]))
        self.baseLength = maxLength
        shiftTable = []
        for i in range(len(points)):
            if points[i] == self.basePoints[0]:
                points = points[i:] + shiftTable
                break
            shiftTable.append(points[i])

        for point in self.contours[0]:
            self.contour.append(point[0])

        shiftTable = []
        for i in range(len(self.contour)):
            if np.all(self.contour[i] == self.basePoints[0]):
                self.contour = self.contour[i:] + shiftTable
                break
            shiftTable.append(self.contour[i])
        self.allPoints = points

        threshold = 0.05

        indx = -1
        self.armsPoints = self.allPoints[indx]
        while abs(self.armsPoints[1] - self.basePoints[0][1]) < self.basePoints[0][1] * threshold:
            indx -= 1
            self.armsPoints = self.allPoints[indx]

        indx = 2
        self.armsPoints = (self.armsPoints, self.allPoints[indx])
        while abs(self.armsPoints[1][1] - self.basePoints[1][1]) < self.basePoints[1][1] * threshold:
            indx += 1
            self.armsPoints = (self.armsPoints[0], self.allPoints[indx])

    def getMetrics(self):
        self.curvePoints = []
        self.contour = []
        self.pointsToCheck = []
        self.allPoints = []
        self.distances = []
        self.basePoints = tuple()

        self.contours, self.hierarchy = cv2.findContours(self.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgCopy = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)

        epsilon = 0.004 * cv2.arcLength(self.contours[0], True)
        approx = cv2.approxPolyDP(self.contours[0], epsilon, True)

        self.getBaseAndShift(approx)
        return approx, imgCopy

    def rotateImage(self, flip="no"):
        if flip == "yes":
            deg = 180
        else:
            deg = math.atan2((self.basePoints[1][1] - self.basePoints[0][1]),
                             (self.basePoints[1][0] - self.basePoints[0][0])) * (180 / math.pi)
            deg %= 180

        (h, w) = self.image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        rotationMatrix = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
        return cv2.warpAffine(self.image, rotationMatrix, (w, h), flags=cv2.INTER_LINEAR)

    def getCurvePointsToCheck(self):
        idx = np.round(np.linspace(0, len(self.curvePoints) - 1, 50)).astype(int)
        # tutaj nie co równą ilość pkt tylko co równą odległość

        for i in idx:
            self.pointsToCheck.append(self.curvePoints[i])

    def getEdgePoints(self):
        curve = False
        for point in self.contour:
            # print(type(point))
            if curve:
                self.curvePoints.append(tuple(point))
            if np.all(point == self.armsPoints[1]):
                self.curvePoints.append(tuple(point))
                curve = True
            if np.all(point == self.armsPoints[0]):
                curve = False

    def getDistances(self):
        for point in self.pointsToCheck:
            self.distances.append((self.basePoints[0][1] - point[1]) / self.baseLength)

    def match_distances(self, piro_objects: List):
        distances = self.distances.copy()
        distances.reverse()

        result_dict = {}

        for o in piro_objects:
            distances2 = o.distances

            sum_list = [sum(x) for x in zip(*[distances, distances2])]
            result_dict[o] = stdev(sum_list)

        final_result = []
        while len(result_dict) > 0:
            best = min(result_dict, key=result_dict.get)
            result_dict.pop(best)
            final_result.append(best.name)

        return final_result


def testMain():
    sets = {"0": 6, "1": 20, "2": 20, "3": 20, "4": 20, "5": 200, '6': 200, '7': 20, '8': 100}
    # sets = {"2": 20}
    finalResults = dict()
    filepath = "../../p1_2/"
    for setKey, setValue in sets.items():
        if setKey in ['7', '8']:
            directory = filepath + "dataB/set"
        else:
            directory = filepath + "dataA/set"

        images = dict()
        for n in range(int(setValue)):
            images[n] = cv2.imread(os.path.join(os.path.dirname(directory + setKey + "/"), str(n) + ".png"),
                                   cv2.IMREAD_GRAYSCALE)
        runCorrectList = []

        with open(directory + setKey + "/" + 'correct.txt', 'r') as f:
            for line in f.readlines():
                runCorrectList.append(int(line))

        img_widths = []
        for im in images.values():
            img_widths.append(im.shape[0])
        avg_width = int(sum(img_widths) / len(images.values()))

        obj_list = []
        for key, im in images.items():
            # scaling
            height = int(avg_width * (im.shape[0] / im.shape[1]))
            im = cv2.resize(im, (avg_width, height))
            piroObject = Piro(im, key)
            piroObject.process()
            obj_list.append(piroObject)

        runResults = []
        for o in obj_list:
            runResults.append(o.match_distances(obj_list))

        runScore = evaluate(runCorrectList, runResults)
        finalResults[setKey] = runScore
        # print("results", runResults)
        print(f"Set{setKey}:", runScore)
    print("results", finalResults)


def runMain():
    images = dict()

    for n in range(int(sys.argv[2])):
        path = Path(sys.argv[1] + "/" + str(n) + ".png")
        images[n] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    img_widths = []
    for im in images.values():
        img_widths.append(im.shape[0])
    avg_width = int(sum(img_widths) / len(images.values()))

    obj_list = []
    for key, im in images.items():
        height = int(avg_width * (im.shape[0] / im.shape[1]))
        im = cv2.resize(im, (avg_width, height))
        piroObject = Piro(im, key)
        piroObject.process()
        obj_list.append(piroObject)

    for o in obj_list:
        print(*o.match_distances(obj_list))


if __name__ == '__main__':
    runMain()
