#!/usr/bin/env python3

import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from process_matches import F as F_ours

def cp_mat(t):
    Tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return Tx

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),10,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),10,color,-1)
    return img1,img2

current_image = cv2.imread('/home/powerhorse/Documents/visualAlignmentComparison/1638558429.333589/cameraimage_0020.jpg')
align_image = cv2.imread('/home/powerhorse/Documents/visualAlignmentComparison/1638558429.333589/alignimage_0020.jpg')

with open('/home/powerhorse/paul_tmp/VisualMatching/output/1638558429.333589/0020/0/results.json') as f:
    data = json.load(f)

align_pose = np.array(data['alignPoseRowMajor']).reshape((4,4))
align_intrinsics = np.array(data['alignIntrinsicsRowMajor']).reshape((3,3))
current_pose = np.array(data['cameraPoseRowMajor']).reshape((4,4))
current_intrinsics = np.array(data['cameraIntrinsicsRowMajor']).reshape((3,3))

current_image_pixel_coordinates = np.array(data['image2MatchCoordinates'])
current_image_pixel_coordinates_homogeneous = np.hstack((current_image_pixel_coordinates, np.ones((current_image_pixel_coordinates.shape[0], 1))))
current_normalized_coordinates_homogeneous = np.matmul(np.linalg.inv(current_intrinsics), current_image_pixel_coordinates_homogeneous.transpose())

align_image_pixel_coordinates = np.array(data['image1MatchCoordinates'])
align_image_pixel_coordinates_homogeneous = np.hstack((align_image_pixel_coordinates, np.ones((align_image_pixel_coordinates.shape[0], 1))))
align_normalized_coordinates_homogeneous = np.matmul(np.linalg.inv(align_intrinsics), align_image_pixel_coordinates_homogeneous.transpose())

ARKitToCamera = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]])

align_pose_camera_convention = np.matmul(align_pose, ARKitToCamera)
current_pose_camera_convention = np.matmul(current_pose, ARKitToCamera)

# why is it like this?!?
current_pose_relative_to_align_pose = np.matmul(np.linalg.inv(align_pose_camera_convention), current_pose_camera_convention)

R = current_pose_relative_to_align_pose[:3,:3]
t = current_pose_relative_to_align_pose[:-1,-1]

#R = current_pose_relative_to_align_pose[:3,:3].transpose()
#t = align_pose_camera_convention[:-1,-1] - current_pose_camera_convention[:-1,-1] 
Tx = cp_mat(t)
# TODO: warning swapping order based on conflicting resources
E_ours = np.matmul(Tx, R)
F_ours = np.matmul(np.matmul(np.linalg.inv(current_intrinsics.transpose()), E_ours), np.linalg.inv(align_intrinsics))

sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(align_image,None)
kp2, des2 = sift.detectAndCompute(current_image,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.6*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


pts2 = np.array(data['image2MatchCoordinates'])
pts1 = np.array(data['image1MatchCoordinates'])
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
E = np.matmul(np.matmul(current_intrinsics.transpose(), F), align_intrinsics)
print("E", E)
print(mask.sum(), mask.shape)
print(F, F_ours)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
print(pts1.shape)
pts1 = pts1[0:10,:]
pts2 = pts2[0:10,:]
print("pts1",pts1)
print("pts2",pts2)
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F_ours)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(align_image,current_image,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F_ours)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(current_image,align_image,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
