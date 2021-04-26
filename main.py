import math
import os
import random
import sys
from statistics import mean
from typing import List, Tuple

import cv2
import numpy as np


def angle(points):
    ang = math.degrees(math.atan2(points[2][1] - points[1][1], points[2][0] - points[1][0]) - math.atan2(
        points[0][1] - points[1][1], points[0][0] - points[1][0]))

    if ang < 0:
        return points[0], points[1], points[2], ang + 360
    return points[0], points[1], points[2], ang


def get_edges(points, base) -> (Tuple, List):
    edges = []
    for i in range(0, len(points) - 1):
        edges.append((points[i], points[i + 1], math.dist(points[i], points[i + 1])))
    edges.append((points[-1], points[0], math.dist(points[-1], points[0])))

    # longest_trinity = []
    # trinity = [edges[0], edges[1]]
    # max_sum = 0
    # for e in edges[2:] + [edges[0], edges[1]]:
    #     trinity.append(e)
    #     actual_sum = trinity[0][2] + trinity[1][2] + trinity[2][2]
    #
    #     if actual_sum > max_sum:
    #         longest_trinity = trinity
    #         max_sum = actual_sum
    #
    #     trinity = trinity[1:]

    # shift edges
    # base = longest_trinity[1]
    base_index = edges.index((base[0], base[1], math.dist(base[0], base[1])))
    edges = edges[base_index:] + edges[:base_index]

    return edges[1:]


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


def distanceBetweenPointAndLine(points):
    # Punkty na linii
    p1 = np.array(points[0])
    p2 = np.array(points[1])

    # Punkt poza linią
    p3 = np.array(points[2])
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


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


def get_integral_points_from_line(point_a: Tuple, point_b: Tuple) -> List[Tuple]:
    # print(point_a,point_b)
    a = (point_b[1] - point_b[0]) / (point_a[1] - point_a[0])
    b = point_b[0] - (a * point_a[0])

    if point_a[0] > point_b[0]:
        start = point_b[0]
        end = point_a[0]
    else:
        start = point_a[0]
        end = point_b[0]

    points = []
    for i in range(start, end + 1):
        y = (a * i) + b
        points.append((i, round(y)))  # TODO: to zaokrąglanie jest do dupy bo tracisz precyzje
    return points


def monte_dupa(base, edges, n_measures):
    p1, p2 = base
    points = get_integral_points_from_line(p1, p2)
    # print("points in base", points)
    # równianie prostej
    a = (p2[1] - p2[0]) / (p1[1] - p1[0])
    # b = p2[0] - (a * p1[0])

    # prostopadła
    a = -1 / a

    edge_points = []
    for e in edges:
        # edge_points.extend(list(zip(np.linspace(e[0][0], e[1][0], 1), #To miało zrobić robotę, ale daje floatowe wymiary
        #                             np.linspace(e[0][1], e[1][1], 1))))
        edge_points.extend(get_integral_points_from_line(e[0], e[1]))

    # print("points in edges", edge_points)

    base_length = math.dist(p1, p2)
    distances = []
    for p in points:
        b = p[1] - (a * p[0])
        for p_e in edge_points:
            if p_e[1] == a * p_e[0] + b:  # TODO: tu nie znajduje idalnych przecięć, chyba też trzeba zaokrąglić
                distances.append(math.dist(p, p_e) / base_length)

    print("distances", distances)
    return distances


class Piro:
    def __init__(self, image, name):
        self.name = name
        self.image = image
        self.basePoints = tuple()
        self.baseLength = 1
        self.armPoints1 = tuple()
        self.armPoints2 = tuple()
        self.allPoints = []
        self.distancesToNearestArmNormalized = dict()
        self.calculatedAngles = dict()
        self.distances = []

    def process(self):
        contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgCopy = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2BGR)

        epsilon = 0.005 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        points = []

        for i in range(-2, len(approx) - 2):
            self.allPoints.append((approx[i][0][0], approx[i][0][1]))
            points.append(((approx[i][0][0], approx[i][0][1]), (approx[i + 1][0][0], approx[i + 1][0][1]),
                           (approx[i + 2][0][0], approx[i + 2][0][1])))

        # base, edges = get_edges(list(self.allPoints))
        # print("edges", base, edges)
        # cv2.line(imgCopy, base[0], base[1], (0, 0, 255), thickness=2)
        # for e in edges:
        #     cv2.line(imgCopy, e[0], e[1], (0, 255, 0), thickness=1)

        angles = []
        for p in points:
            angles.append(angle(p))
        shiftedAngles = shiftAngleArray(angles)
        # print(shiftedAngles)
        self.makeCalculatedAngles(shiftedAngles)
        # print(self.calculatedAngles)
        self.basePoints = shiftedAngles[0][1], shiftedAngles[0][2]

        # Tu jest wywołanie tych krawędzi
        edges = get_edges(self.allPoints, self.basePoints)
        self.distances = monte_dupa(self.basePoints, edges, 10)

        self.armPoints1 = (shiftedAngles[0][0], shiftedAngles[0][1])
        self.armPoints2 = (shiftedAngles[1][1], shiftedAngles[1][2])
        self.allPoints.remove(self.basePoints[0])
        self.allPoints.remove(self.basePoints[1])
        self.baseLength = math.dist(self.basePoints[0], self.basePoints[1])

        for p in self.allPoints:
            self.distancesToNearestArmNormalized[p] = (min(
                distanceBetweenPointAndLine((self.armPoints1[0], self.armPoints1[1], p)),
                distanceBetweenPointAndLine((self.armPoints2[0], self.armPoints2[1], p))) / self.baseLength)
            distanceBetweenPointAndLine((p, self.basePoints[0], self.basePoints[1]))

        # print("d", self.distancesToNearestArmNormalized)

        # cv2.drawContours(imgCopy, [approx], 0, color=(0, 0, 255), thickness=2)
        # cv2.imshow("image", imgCopy)
        # cv2.waitKey()

    def match_distances(self, piroObjects: List):
        distances = self.distances.copy()
        distances.reverse()

        result_dict = {}

        for o in piroObjects:
            distances2 = o.distances

            if len(distances) > len(
                    distances2):  # TODO: to jest smrut, raczej wszystkie dystanse powinny być takiej samej długości
                distances = distances[:len(distances2)]
            elif len(distances) < len(distances2):
                distances2 = distances2[:len(distances)]

            avg = distances[0] + distances2[0]
            avg_diff = []
            for i in range(1, len(distances)):
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

    def makeCalculatedAngles(self, angles):
        for a in angles:
            self.calculatedAngles[a[1]] = a[3]


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
        piroObject.process()
        obj_list.append(piroObject)

    results = []
    for o in obj_list:
        results.append(o.match_distances(obj_list))

    print("results", results)
    print("Final score", evaluate(correct_list, results))
