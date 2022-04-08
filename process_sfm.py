#!/usr/bin/env python3

# SEE: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html

import cv2
import json
import numpy as np

with open('output_sfm/results.json') as f:
    d = json.load(f)

K1 = np.array(d['intrinsics1RowMajor']).reshape((3,3))
K2 = np.array(d['intrinsics2RowMajor']).reshape((3,3))

P1 = np.array(d['pose1RowMajor']).reshape((4,4))
P2 = np.array(d['pose2RowMajor']).reshape((4,4))

# dropMat = [1 0 0 0;
#            0 1 0 0;
#            0 0 1 0]
dropMat = np.hstack((np.eye(3,3), np.zeros((3,1))))

deviceToImage = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

projectionMatrix1 = np.matmul(K1, np.matmul(dropMat, np.linalg.inv(np.matmul(P1, deviceToImage))))
projectionMatrix2 = np.matmul(K2, np.matmul(dropMat, np.linalg.inv(np.matmul(P2, deviceToImage))))

image1MatchCoordinates = np.array(d['image1MatchCoordinates']).transpose()
image2MatchCoordinates = np.array(d['image2MatchCoordinates']).transpose()

homogeneous = cv2.triangulatePoints(projectionMatrix1, projectionMatrix2, image1MatchCoordinates, image2MatchCoordinates)
inhomogeneous = cv2.convertPointsFromHomogeneous(homogeneous.transpose()).squeeze()

im1Reconstructed = cv2.convertPointsFromHomogeneous(np.matmul(projectionMatrix1, homogeneous).transpose()).squeeze()
im2Reconstructed = cv2.convertPointsFromHomogeneous(np.matmul(projectionMatrix2, homogeneous).transpose()).squeeze()

print("im1 reprojection errors", im1Reconstructed - image1MatchCoordinates.transpose())
print("im2 reprojection errors", im2Reconstructed - image2MatchCoordinates.transpose())
