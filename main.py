import math
import os
import sys
from statistics import mean
from typing import List
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
        self.image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
        self.basePoints = tuple()
        self.baseLength = 1
        self.allPoints = []
        self.distances = []
        self.pointsToCheck = []
        self.contours = None
        self.hierarchy = None

    def process(self):
        _ = self.getMetrics()
        self.image = self.rotateImage()
        approx, imgCopy = self.getMetrics()
        if self.basePoints[0][1] < self.image.shape[:2][0] / 2:
            self.image = self.rotateImage(flip="yes")
            approx, imgCopy = self.getMetrics()

        self.getCurvePointsToCheck(2)

        # Tu jest wywołanie tych krawędzi
        # edges = get_edges(self.allPoints, self.basePoints)
        # self.distances = monte_dupa(self.basePoints, edges, 10)

        self.getDistances()

        # cv2.drawContours(imgCopy, [approx], 0, color=(0, 0, 255), thickness=2)

        for point in self.pointsToCheck:
            cv2.circle(imgCopy, point, radius=0, color=(0, 255, 0), thickness=6)

        print(self.basePoints)
        cv2.imshow("image", imgCopy)
        cv2.waitKey()

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
        self.allPoints = points

    def getMetrics(self):
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

    def getCurvePointsToCheck(self, threshold):
        arms = (self.allPoints[-1], self.allPoints[2])
        armDistance = arms[1][0] - arms[0][0]
        step = round(armDistance * (threshold / 100))
        for i in range(0, round(armDistance), step):
            for p in self.contours[0]:
                if p[0][0] == (arms[0][0] + i) and p[0][1] not in range(self.basePoints[0][1] - 5,
                                                                        self.basePoints[0][1] + 5):
                    self.pointsToCheck.append((arms[0][0] + i, p[0][1]))
                    break

    def getDistances(self):
        for point in self.pointsToCheck:
            self.distances.append((self.basePoints[0][1] - point[1]) / self.baseLength)

    def match_distances(self, piroObjects: List):
        distances = self.distances.copy()
        distances.reverse()

        result_dict = {}

        for o in piroObjects:
            distances2 = o.distances

            if len(distances) > len(distances2):  # TODO: to jest smrut, raczej wszystkie dystanse powinny być takiej samej długości
                distances = distances[:len(distances2)]
            elif len(distances) < len(distances2):
                distances2 = distances2[:len(distances)]

            avg = distances[0] + distances2[0]
            avg_diff = []
            for i in range(len(distances)):
                avg_diff.append(abs(avg - (distances[i] + distances2[i])))
            print("avg_diff", avg_diff)
            result_dict[o] = mean(avg_diff)

        final_result = []
        while len(result_dict) > 0:
            best = min(result_dict, key=result_dict.get)
            result_dict.pop(best)
            final_result.append(best.name)

        print("results for image", final_result)
        return final_result

    def solve(self) -> List:
        return [0]


if __name__ == '__main__':
    images = dict()
    for n in range(int(sys.argv[2])):
        images[n] = cv2.imread(os.path.join(os.path.dirname(sys.argv[1]), str(n) + ".png"), cv2.IMREAD_GRAYSCALE)
    correct_list = []

    with open(sys.argv[1] + 'correct.txt', 'r') as f:
        for line in f.readlines():
            correct_list.append(int(line))

    obj_list = []
    for key, im in images.items():
        piroObject = Piro(im, key)
        print(key)
        piroObject.process()
        obj_list.append(piroObject)

    results = []
    for o in obj_list:
        results.append(o.match_distances(obj_list))

    print("results", results)
    print("Final score", evaluate(correct_list, results))
