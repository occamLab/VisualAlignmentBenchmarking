#!/usr/bin/env python3

import numpy as np
import cv2
import json

def drawlines(img1,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
    return img1,img2

def mouse_event(event,x,y,flag,im):
    x = x * downsample_factor
    y = y * downsample_factor
    """ Catch mouse events so we can draw epipolar lines when clicked """
    if event == cv2.EVENT_FLAG_LBUTTON:
        if x < im.shape[1]*downsample_factor/2.0:
            #l = cv2.computeCorrespondEpilines(np.array([[x,y]]), 2, F)
            print("on left side")
            l = F_alt_1.dot(np.array([x,y,1.0]))
            #m = -l[0][0][0]/l[0][0][1]
            #b = -l[0][0][2]/l[0][0][1]
            m = -l[0]/l[1]
            b = -l[2]/l[1]
            # equation of the line is y = m*x+b
            y_for_x_min = m*0.0+b
            y_for_x_max = m*(im.shape[1]*downsample_factor/2.0-1)+b
            print(m, b, y_for_x_min, y_for_x_max)
            # plot the epipolar line
            cv2.line(im,(int(im.shape[1]/2.0),int(y_for_x_min)),(int(im.shape[1]-1.0),int(y_for_x_max)),(255,0,0))



with open('/home/powerhorse/paul_tmp/VisualMatching/output/1638558429.333589/0023/0/results.json') as f:
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

current_pose_relative_to_align_pose = np.matmul(np.linalg.inv(align_pose_camera_convention), current_pose_camera_convention)

R = current_pose_relative_to_align_pose[:3,:3]
t = current_pose_relative_to_align_pose[:3,-1] 

Tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])

E = np.matmul(R, Tx)
(F_alt_1,mask_1) = cv2.findFundamentalMat(np.int32(current_image_pixel_coordinates), np.int32(align_image_pixel_coordinates))
(F_alt_2,mask_2) = cv2.findFundamentalMat(np.int32(align_image_pixel_coordinates), np.int32(current_image_pixel_coordinates))
F = np.matmul(np.matmul(np.linalg.inv(align_intrinsics.transpose()), E), np.linalg.inv(current_intrinsics))
#F_alt_1 = np.matmul(np.matmul(np.linalg.inv(current_intrinsics.transpose()), E_alt_1), np.linalg.inv(align_intrinsics))
#F_alt_2 = np.matmul(np.matmul(np.linalg.inv(align_intrinsics.transpose()), E_alt_2), np.linalg.inv(current_intrinsics))
#F_alt_3 = np.matmul(np.matmul(np.linalg.inv(align_intrinsics.transpose()), E_alt_1), np.linalg.inv(current_intrinsics))
#F_alt_4 = np.matmul(np.matmul(np.linalg.inv(current_intrinsics.transpose()), E_alt_2), np.linalg.inv(align_intrinsics))

# find Epipolar matrix using OpenCV and use that to plot epipoles

epipolar_constraint = []
for i in range(current_normalized_coordinates_homogeneous.shape[1]):
    # Note: we think it sould be align'*E*current but current'*E*align seems to agree with OpenCV better epipolar_constraint.append(np.matmul(current_normalized_coordinates_homogeneous[:,i], np.matmul(E, align_normalized_coordinates_homogeneous[:,i])))

epipolar_constraint = np.array(epipolar_constraint)
print(epipolar_constraint)

current_image = cv2.imread('/home/powerhorse/Documents/visualAlignmentComparison/1638558429.333589/cameraimage_0023.jpg')
align_image = cv2.imread('/home/powerhorse/Documents/visualAlignmentComparison/1638558429.333589/alignimage_0023.jpg')

best_match_index = np.argmin(np.abs(epipolar_constraint))

print(np.sum(np.abs(epipolar_constraint)))
for i in range(len(epipolar_constraint)):
    if np.abs(epipolar_constraint[i]) < np.quantile(np.abs(epipolar_constraint), 0.25):
        color = (0, 255, 0)
    elif np.abs(epipolar_constraint[i]) > np.quantile(np.abs(epipolar_constraint), 0.75):
        color = (0, 0, 255)
    else:
        color = None

    if color is not None:
        align_center = (int(align_image_pixel_coordinates[i,0]), int(align_image_pixel_coordinates[i,1]))
        cv2.circle(align_image, align_center,5, color, -1)

        current_center = (int(current_image_pixel_coordinates[i,0]), int(current_image_pixel_coordinates[i,1]))
        cv2.circle(current_image, current_center,5, color, -1)

downsample_factor = 2
merged = np.hstack((align_image, current_image))
merged_resize = cv2.resize(merged, (merged.shape[1]//downsample_factor, merged.shape[0]//downsample_factor))
#cv2.imshow('merged_resized', merged_resize)
#cv2.setMouseCallback('merged_resized',mouse_event,merged_resize)
#for i in range(10000):
#    cv2.imshow('merged_resized', merged_resize)
#    cv2.waitKey(1)
