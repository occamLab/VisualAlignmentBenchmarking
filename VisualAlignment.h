//
//  VisualAlignment.h
//  Clew
//
//  Created by Kawin Nikomborirak on 7/9/19.
//  Copyright © 2019 OccamLab. All rights reserved.
//

#ifndef visual_alignment_h
#define visual_alignment_h

#include <opencv2/opencv.hpp>
#include "VisualAlignmentUtils.hpp"

typedef struct {
    float yaw;
    bool is_valid;
    int numInliers;
    int numMatches;
    float residualAngle;
    float tx;
    float ty;
    float tz;
    cv::Mat matchingImage;
    std::vector<std::vector<float> > triangulatedPoints;
    std::vector<std::vector<float> > image1MatchCoordinates;
    std::vector<std::vector<float> > image2MatchCoordinates;
} VisualAlignmentReturn;

/**
 Deduce the yaw between two images.
 
 - returns: The yaw in radians between the pictures assuming portrait orientation.
 
 - parameters:
 - image1: The image the returned yaw is relative to.
 - intrinsics1: The camera intrinsics used to take image1 in the format [fx, fy, ppx, ppy].
 - pose1: The pose of the camera in the arsession used to take the first image.
 - image2: The image the returned yaw rotates to.
 - intrinsics2: The camera intrinsics used to take image2 in the format [fx, fy, ppx, ppy].
 - pose2: The pose of the camera in the arsession used to take the second image.
 */

VisualAlignmentReturn visualYaw(cv::Mat * image1,  cv::Mat* intrinsics1, cv::Mat* pose1, cv::Mat* image2, cv::Mat* intrinsics2, cv::Mat*  pose2,  int downSampleFactor, VisualFeatureType vftype, double ratioTest, bool debug);
#endif
