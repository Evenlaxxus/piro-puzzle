import math
import os
import sys
import cv2


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


# def scale_to_one_size
# def rotate(image, base_points, contours):
#     p1, p2 = base_points
#     start_point, end_point = p1[2], p2[0]
#     angle = math.atan2((end_point[1] - start_point[1]), (end_point[0] - start_point[0])) * (180.0 / math.pi)
#     angle %= 180.0
#
#     moments = cv2.moments(contours[0])
#     center_x = int(moments["m10"] / moments["m00"])
#     center_y = int(moments["m01"] / moments["m00"])
#     rot_mat = cv2.getRotationMatrix2D((center_y, center_x), angle, 1.0)
#
#     height, width = image.shape
#     res_img = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
#
#
#     contours, hierarchy = cv2.findContours(res_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     imgCopy = cv2.cvtColor(res_img.copy(), cv2.COLOR_GRAY2BGR)
#
#     epsilon = 0.005 * cv2.arcLength(contours[0], True)
#     approx = cv2.approxPolyDP(contours[0], epsilon, True)
#
#     cv2.drawContours(imgCopy, [approx], 0, color=(0, 0, 255), thickness=2)
#     cv2.imshow("eee", imgCopy)
#
#     return res_img


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
            cv2.line(imgCopy, base_points[0], base_points[1], (0, 255, 0))

            cv2.drawContours(imgCopy, [approx], 0, color=(0, 0, 255), thickness=2)
            cv2.imshow("image", imgCopy)
            cv2.waitKey()


if __name__ == '__main__':
    piroObject = Piro()
    piroObject.load(os.path.dirname(sys.argv[1]), int(sys.argv[2]))
    piroObject.solve()
