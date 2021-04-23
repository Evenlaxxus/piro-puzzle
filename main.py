import math
import os
import sys
import cv2 as cv
import numpy as np


class piro:
    def __init__(self):
        self.images = dict()

    def load(self, directory, numberOfImages):
        for n in range(numberOfImages):
            self.images[n] = cv.imread(os.path.join(directory, str(n) + ".png"), cv.IMREAD_GRAYSCALE)

    def solve(self):
        for key, im in self.images.items():
            contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            imgCopy = cv.cvtColor(im.copy(), cv.COLOR_GRAY2BGR)

            epsilon = 0.007 * cv.arcLength(contours[0], True)
            approx = cv.approxPolyDP(contours[0], epsilon, True)
            print(len(approx))
            vectors = []

            for i in range(len(approx) - 1):
                vectors.append((approx[i + 1, 0, 0] - approx[i, 0, 0], approx[i + 1, 0, 1] - approx[i, 0, 1]))
            vectors.append((approx[-1, 0, 0] - approx[0, 0, 0], approx[-1, 0, 1] - approx[0, 0, 1]))
            print(vectors)

            angles = []
            for v in range(len(vectors) - 1):
                angles.append(self.angle(vectors[v], vectors[v + 1]))
            angles.append(self.angle(vectors[-1], vectors[0]))
            print(angles)

            cv.drawContours(imgCopy, [approx], 0, color=(255, 0, 0), thickness=2)
            cv.imshow("image", imgCopy)
            cv.waitKey()

    def angle(self, v1, v2):
        unit_vector_1 = v1 / np.linalg.norm(v1)
        unit_vector_2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        return np.arccos(dot_product) * (180.0 / math.pi)


if __name__ == '__main__':
    piroObject = piro()
    piroObject.load(os.path.dirname(sys.argv[1]), 6)
    piroObject.solve()
