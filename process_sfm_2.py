#!/usr/bin/env python3

# SEE: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html

import cv2
import json
import numpy as np

with open('output_sfm/results.json') as f:
    d = json.load(f)

K1 = np.array(d['intrinsics1RowMajor']).reshape((3,3))
K2 = np.array(d['intrinsics2RowMajor']).reshape((3,3))

deviceToImage = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

P1 = np.matmul(np.array(d['pose1RowMajor']).reshape((4,4)), deviceToImage)
P2 = np.matmul(np.array(d['pose2RowMajor']).reshape((4,4)), deviceToImage)
print(P1)
print(P2)
# dropMat = [1 0 0 0;
#            0 1 0 0;
#            0 0 1 0]
dropMat = np.hstack((np.eye(3,3), np.zeros((3,1))))
normalizedImage1MatchCoordinates = np.matmul(np.linalg.inv(K1), cv2.convertPointsToHomogeneous(np.array(d['image1MatchCoordinates'])).squeeze().transpose())
normalizedImage2MatchCoordinates = np.matmul(np.linalg.inv(K2), cv2.convertPointsToHomogeneous(np.array(d['image2MatchCoordinates'])).squeeze().transpose())

nPts1 = cv2.convertPointsFromHomogeneous(normalizedImage1MatchCoordinates.transpose()).squeeze()
nPts2 = cv2.convertPointsFromHomogeneous(normalizedImage2MatchCoordinates.transpose()).squeeze()
print(nPts1[0,:])
print(nPts2[0,:])
projectionMatrix1 = np.matmul(dropMat, np.linalg.inv(P1))
projectionMatrix2 = np.matmul(dropMat, np.linalg.inv(P2))

homogeneous = cv2.triangulatePoints(projectionMatrix1, projectionMatrix2, nPts1.transpose(), nPts2.transpose())
inhomogeneous = cv2.convertPointsFromHomogeneous(homogeneous.transpose()).squeeze()
print(inhomogeneous[0,:])
im1Reconstructed = cv2.convertPointsFromHomogeneous(np.matmul(projectionMatrix1, homogeneous).transpose()).squeeze()
im2Reconstructed = cv2.convertPointsFromHomogeneous(np.matmul(projectionMatrix2, homogeneous).transpose()).squeeze()
