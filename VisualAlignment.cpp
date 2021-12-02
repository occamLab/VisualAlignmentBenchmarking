//
//  VisualAlignment.mm
//  Clew
//
//  Created by Kawin Nikomborirak on 7/9/19.
//  Copyright © 2019 OccamLab. All rights reserved.
//

#include "VisualAlignment.h"
#include "VisualAlignmentUtils.hpp"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "json.hpp"
#include <fstream>
#include <sys/stat.h>

using json = nlohmann::json;
namespace po = boost::program_options;

VisualAlignmentReturn visualYaw(cv::Mat * image1,  cv::Mat* intrinsics1, cv::Mat* pose1, cv::Mat* image2, cv::Mat* intrinsics2, cv::Mat*  pose2,  int downSampleFactor, VisualFeatureType vftype, double ratioTest, bool debug) {
    bool useThreePoint = true;
    VisualAlignmentReturn ret;
    cv::Mat image_mat1, image_mat2;
    cv::cvtColor(*image1, image_mat1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(*image2, image_mat2, cv::COLOR_BGR2GRAY);
    cv::Mat image_mat1_orig = image_mat1;
    cv::Mat image_mat2_orig = image_mat2;
    cv::rotate(image_mat1, image_mat1, cv::ROTATE_90_CLOCKWISE);
    cv::rotate(image_mat2, image_mat2, cv::ROTATE_90_CLOCKWISE);
  
    if (debug) {
        cv::namedWindow("image1_rotated");
        cv::namedWindow("image2_rotated");
        cv::imshow("image1_rotated", image_mat1);
        cv::imshow("image2_rotated", image_mat2);
    }
    Eigen::Matrix3f intrinsics1_matrix_unrotated = intrinsicsToMatrix(intrinsics1);
    Eigen::Matrix3f intrinsics2_matrix_unrotated = intrinsicsToMatrix(intrinsics2);
    
    // Since the image was rotated clockwise, we have to swap entries in the intrinsic matrices as well.
    // I use matrix multiplication for this.
    Eigen::Matrix3f swap_matrix;
    swap_matrix << 0, 1, 0, 1, 0, 0, 0, 0, 1;
    
    Eigen::Matrix3f intrinsics1_matrix = swap_matrix * intrinsics1_matrix_unrotated * swap_matrix;
    Eigen::Matrix3f intrinsics2_matrix = swap_matrix * intrinsics2_matrix_unrotated * swap_matrix;
    
    intrinsics1_matrix(0, 2) = image_mat1.cols - intrinsics1_matrix(0, 2);
    intrinsics2_matrix(0, 2) = image_mat2.cols - intrinsics2_matrix(0, 2);
    const Eigen::Matrix4f pose1_matrix = poseToMatrix(pose1);
    const Eigen::Matrix4f pose2_matrix = poseToMatrix(pose2);
    
    const Eigen::AngleAxisf square_rotation1 = getIdealRotation(pose1_matrix);
    const Eigen::AngleAxisf square_rotation2 = getIdealRotation(pose2_matrix);
    
    const auto square_image_mat1 = warpPerspectiveWithGlobalRotation(image_mat1, intrinsics1_matrix, pose1_matrix.block(0, 0, 3, 3), square_rotation1);
    const auto square_image_mat2 = warpPerspectiveWithGlobalRotation(image_mat2, intrinsics2_matrix, pose2_matrix.block(0, 0, 3, 3), square_rotation2);
    cv::Mat homography_mat_for_dewarping_1 = getHomographyForGlobalRotation(intrinsics1_matrix, pose1_matrix.block(0, 0, 3, 3), square_rotation1).inv();
    cv::Mat homography_mat_for_dewarping_2 = getHomographyForGlobalRotation(intrinsics2_matrix, pose2_matrix.block(0, 0, 3, 3), square_rotation2).inv();

    cv::Mat square_image_mat1_resized, square_image_mat2_resized;
    cv::resize(square_image_mat1, square_image_mat1_resized, cv::Size(square_image_mat1.size().width/downSampleFactor, square_image_mat1.size().height/downSampleFactor));
    cv::resize(square_image_mat2, square_image_mat2_resized, cv::Size(square_image_mat2.size().width/downSampleFactor, square_image_mat2.size().height/downSampleFactor));
    
    auto keypoints_and_descriptors1 = getKeyPointsAndDescriptors(square_image_mat1_resized, vftype);
    auto keypoints_and_descriptors2 = getKeyPointsAndDescriptors(square_image_mat2_resized, vftype);

    if (debug) {
        cv::namedWindow("image1_warped");
        cv::namedWindow("image2_warped");
        cv::imshow("image1_warped", square_image_mat1_resized);
        cv::imshow("image2_warped", square_image_mat2_resized);
    }

    const auto matches = getMatches(keypoints_and_descriptors1.descriptors, keypoints_and_descriptors2.descriptors, ratioTest);
    
    
    std::vector<cv::Point2f> vectors1, vectors2;
  
    // dewarped points
    auto keypoints1_unwarped = keypoints_and_descriptors1.keypoints;
    auto keypoints2_unwarped = keypoints_and_descriptors2.keypoints;
    // dewarp
    cv::Matx33f warp1 = homography_mat_for_dewarping_1;
    for (auto& keypoint : keypoints1_unwarped) {
        cv::Point2f warped_point(downSampleFactor*keypoint.pt.x, downSampleFactor*keypoint.pt.y);
        cv::Point3f unwarped_point = warp1 * warped_point;
        unwarped_point /= unwarped_point.z;
        // this undoes the 90 degree clockwise rotation of the image
        keypoint.pt.x = unwarped_point.y;
        keypoint.pt.y = -unwarped_point.x + image_mat1_orig.size().height;
    }
    // dewarp
    cv::Matx33f warp2 = homography_mat_for_dewarping_2;
    for (auto& keypoint : keypoints2_unwarped) {
        cv::Point2f warped_point(downSampleFactor*keypoint.pt.x, downSampleFactor*keypoint.pt.y);
        cv::Point3f unwarped_point = warp2 * warped_point;
        unwarped_point /= unwarped_point.z;
        // this undoes the 90 degree clockwise rotation of the image
        keypoint.pt.x = unwarped_point.y;
        keypoint.pt.y = -unwarped_point.x + image_mat2_orig.size().height;
    }

    for (const auto& match : matches) {
        auto unwarpedKeypoint1 = keypoints1_unwarped[match.queryIdx];
        auto unwarpedKeypoint2 = keypoints2_unwarped[match.trainIdx];
        auto keypoint1 = keypoints_and_descriptors1.keypoints[match.queryIdx];
        auto keypoint2 = keypoints_and_descriptors2.keypoints[match.trainIdx];

        // TODO: Note: these have been rotated
        std::vector<float> keypoint1ForReturn, keypoint2ForReturn;
        keypoint1ForReturn.push_back(unwarpedKeypoint1.pt.x);
        keypoint1ForReturn.push_back(unwarpedKeypoint1.pt.y);
        keypoint2ForReturn.push_back(unwarpedKeypoint2.pt.x);
        keypoint2ForReturn.push_back(unwarpedKeypoint2.pt.y);
        ret.image1MatchCoordinates.push_back(keypoint1ForReturn);
        ret.image2MatchCoordinates.push_back(keypoint2ForReturn);

        // correct for the downsampling
        vectors1.push_back(downSampleFactor*keypoint1.pt);
        
        // Convert the second keypoint to one with the intrinsics of the first camera.
        Eigen::Vector3f keypoint2vec;
        keypoint2vec << downSampleFactor*keypoint2.pt.x, downSampleFactor*keypoint2.pt.y, 1;
        Eigen::Vector3f keypoint2projected = intrinsics1_matrix * intrinsics2_matrix.inverse() * keypoint2vec;
        vectors2.push_back(cv::Point2f(keypoint2projected(0), keypoint2projected(1)));
    }
   

    if (debug) {
        cv::drawMatches(image_mat1_orig, keypoints1_unwarped, image_mat2_orig, keypoints2_unwarped, matches, ret.matchingImage);
        cv::Mat resized_matching_image;
        cv::resize(ret.matchingImage, resized_matching_image, cv::Size(ret.matchingImage.size().width/downSampleFactor, ret.matchingImage.size().height/downSampleFactor));
        cv::namedWindow("matched_orig");
        cv::imshow("matched_orig", resized_matching_image);
        cv::waitKey(0);
    }
    if (useThreePoint) {
        ret.numMatches = matches.size();
        if (matches.size() < 6) {
            std::cerr << "Images could not be aligned (<6 good matches)!" << std::endl;
            ret.is_valid = false;
            ret.yaw = 0;
            return ret;
        }
        std::vector<Eigen::Vector3d> all_rays_image_1, all_rays_image_2;

        for (unsigned int i = 0; i < matches.size(); i++) {
            const auto& match = matches[i];
            const auto keypoint1 = keypoints_and_descriptors1.keypoints[match.queryIdx];
            const auto keypoint2 = keypoints_and_descriptors2.keypoints[match.trainIdx];
            // correct for the downsampling
            Eigen::Vector3f homogeneousKp1(downSampleFactor*keypoint1.pt.x, downSampleFactor*keypoint1.pt.y, 1.0);
            Eigen::Vector3f image_1_ray = intrinsics1_matrix.inverse() * homogeneousKp1;
            all_rays_image_1.push_back(Eigen::Vector3d(image_1_ray.x(), image_1_ray.y(), image_1_ray.z()));
            Eigen::Vector3f homogeneousKp2(downSampleFactor*keypoint2.pt.x, downSampleFactor*keypoint2.pt.y, 1.0);
            Eigen::Vector3f image_2_ray = intrinsics2_matrix.inverse() * homogeneousKp2;
            all_rays_image_2.push_back(Eigen::Vector3d(image_2_ray.x(), image_2_ray.y(), image_2_ray.z()));
        }
        
        // We'll do RANSAC to find the best three points
        Eigen::Matrix3d bestEssential;
        int bestInlierCount = -1;
        double bestInlierResidualSum = -1;

        // to allow us to randomize for RANSAC
        unsigned int* indices = new unsigned int[matches.size()];
        for (unsigned int i = 0; i < matches.size(); i++) {
            indices[i] = i;
        }
        std::map<int, std::vector<float> > centiradQuantization;
        std::map<int, std::vector<cv::Mat> > centiradQuantizationTranslations;

        for (unsigned int trial = 0; trial < 100; trial++) {
            std::random_shuffle(indices, indices+matches.size());
            std::vector<cv::Point2f> vectors1_ransac, vectors2_ransac;
            Eigen::Vector3d rotation_axis = Eigen::Vector3d(0, 1, 0);
            Eigen::Vector3d image_1_rays[3];
            Eigen::Vector3d image_2_rays[3];
            std::vector<Eigen::Quaterniond> soln_rotations;
            std::vector<Eigen::Vector3d> soln_translations;
            for (unsigned int i = 0; i < 3; i++) {
                const auto& match = matches[indices[i]];
                const auto keypoint1 = keypoints_and_descriptors1.keypoints[match.queryIdx];
                const auto keypoint2 = keypoints_and_descriptors2.keypoints[match.trainIdx];
                // correct for the downsampling
                Eigen::Vector3f homogeneousKp1(downSampleFactor*keypoint1.pt.x, downSampleFactor*keypoint1.pt.y, 1.0);
                Eigen::Vector3f image_1_ray = intrinsics1_matrix.inverse() * homogeneousKp1;
                image_1_rays[i] = Eigen::Vector3d(image_1_ray.x(), image_1_ray.y(), image_1_ray.z());
                Eigen::Vector3f homogeneousKp2(downSampleFactor*keypoint2.pt.x, downSampleFactor*keypoint2.pt.y, 1.0);
                Eigen::Vector3f image_2_ray = intrinsics2_matrix.inverse() * homogeneousKp2;
                image_2_rays[i] = Eigen::Vector3d(image_2_ray.x(), image_2_ray.y(), image_2_ray.z());
                vectors1_ransac.push_back(vectors1[indices[i]]);
                vectors2_ransac.push_back(vectors2[indices[i]]);
            }

            theia::ThreePointRelativePosePartialRotation(rotation_axis,
                                                  image_1_rays,
                                                  image_2_rays,
                                                  &soln_rotations,
                                                  &soln_translations);
            for (unsigned int i = 0; i < soln_rotations.size(); i++) {
                const Eigen::Matrix3d relative_rotation = soln_rotations[i].toRotationMatrix();
                Eigen::Matrix3d essential_matrix = CrossProductMatrix(soln_translations[i]) * relative_rotation;
                essential_matrix.normalize();
                int totalInliers = 0;
                double inlierResidualSum = 0.0;
                for (unsigned int j = 0; j < all_rays_image_1.size(); j++) {
                    double pointResidual = abs(all_rays_image_2[j].transpose() * essential_matrix * all_rays_image_1[j]);
                    if (pointResidual < 0.001) { // TODO: this threshold is not correct, we need to figure out how to make this into something consistent (e.g., distance in pixels to epipolar line)
                        totalInliers++;
                        inlierResidualSum += pointResidual;
                    }
                }
                
                // TODO this needs to be tuned in a smarter way (e.g., by running some iterations of RANSAC first and then adapting the threshold as a proportion of the best inlier count
                if (totalInliers > 0.5*all_rays_image_1.size()) {
                    // compute pose for averaging purposes
                    cv::Mat essential_matrixCV;
                    eigen2cv(essential_matrix, essential_matrixCV);
                    cv::Mat dcm_mat, translation_mat;

                    int numInliers = cv::recoverPose(essential_matrixCV, vectors1_ransac, vectors2_ransac, dcm_mat, translation_mat, intrinsics1_matrix(0, 0), cv::Point2f(intrinsics1_matrix(0, 2), intrinsics1_matrix(1, 2)));
                    if (numInliers < 3) {
                        // one of the correspondences is behind the camera
                        continue;
                    }
                    Eigen::Matrix3f dcm;
                    cv2eigen(dcm_mat, dcm);
                    const auto rotated = dcm * Eigen::Vector3f::UnitZ();
                    float yaw = atan2(rotated(0), rotated(2));
                    int quantized = (int) (yaw*100);
                    if (centiradQuantization.find(quantized) == centiradQuantization.end()) {
                        centiradQuantization[quantized] = std::vector<float>();
                        centiradQuantizationTranslations[quantized] = std::vector<cv::Mat>();
                    }
                    centiradQuantization[quantized].push_back(yaw);
                    centiradQuantizationTranslations[quantized].push_back(translation_mat);
                }
                if (bestInlierCount < 0 || totalInliers > bestInlierCount || (totalInliers == bestInlierCount && inlierResidualSum < bestInlierResidualSum)) {
                    bestInlierCount = totalInliers;
                    bestEssential = essential_matrix;
                    bestInlierResidualSum = inlierResidualSum;
                }
            }
        }
        
        float bestConsensusYaw = 0.0;
        cv::Mat bestConsensusTranslation = cv::Mat(3,1, CV_64F, 0.0);
        unsigned long mostQuantized = 0;
        for (std::map<int, std::vector<float> >::iterator i = centiradQuantization.begin(); i != centiradQuantization.end(); ++i) {
            if (i->second.size() > mostQuantized) {
                mostQuantized = i->second.size();
                bestConsensusYaw = 0.0;
                // take the average of all elements in the bucket
                for (std::vector<float>::iterator j = i->second.begin(); j != i->second.end(); ++j) {
                    bestConsensusYaw += *j / mostQuantized;
                }
                bestConsensusTranslation = cv::Mat(3,1, CV_64F, 0.0);
                for (unsigned long j = 0; j < centiradQuantizationTranslations[i->first].size(); j++) {
                    bestConsensusTranslation += centiradQuantizationTranslations[i->first][j];
                }
                bestConsensusTranslation = bestConsensusTranslation / cv::norm(bestConsensusTranslation);
            }
            
        }
        
        cv::drawMatches(square_image_mat1_resized, keypoints_and_descriptors1.keypoints, square_image_mat2_resized, keypoints_and_descriptors2.keypoints, matches, ret.matchingImage);
        if (debug) {
            cv::namedWindow("matched");
            cv::imshow("matched", ret.matchingImage);
        }
        if (debug) {
            cv::waitKey(0);
        }
        cv::Mat bestEssentialCV;
        eigen2cv(bestEssential, bestEssentialCV);
        cv::Mat dcm_mat, translation_mat, outlierMask, triangulatedPoints, triangulatedPointsInhomogeneous, intrinsics1Rotated;
        intrinsics1Rotated = (cv::Mat_<float>(3,3) << intrinsics1_matrix(0, 0), 0, intrinsics1_matrix(0, 2), 0, intrinsics1_matrix(1, 1) , intrinsics1_matrix(1, 2) , 0, 0, 1);

        int numInliers = cv::recoverPose(bestEssentialCV, vectors1, vectors2, intrinsics1Rotated, dcm_mat, translation_mat, 50, outlierMask, triangulatedPoints);
        ret.numInliers = numInliers;
        for (int i = 0; i < triangulatedPoints.cols; i++) {
            std::vector<float> triangulatedPoint;
            triangulatedPoint.push_back(triangulatedPoints.at<double>(0,i) / triangulatedPoints.at<double>(3,i));
            triangulatedPoint.push_back(triangulatedPoints.at<double>(1,i) / triangulatedPoints.at<double>(3,i));
            triangulatedPoint.push_back(triangulatedPoints.at<double>(2,i) / triangulatedPoints.at<double>(3,i));
            ret.triangulatedPoints.push_back(triangulatedPoint);
        }
        Eigen::Matrix3f dcm;
        cv2eigen(dcm_mat, dcm);
        const auto rotated = dcm * Eigen::Vector3f::UnitZ();
        const float yaw = atan2(rotated(0), rotated(2));
        float residualAngle = abs(yaw) - acos((dcm.trace() - 1)/2);
        ret.yaw = mostQuantized > 0 ? bestConsensusYaw : yaw;
        ret.residualAngle = residualAngle;
        ret.tx = mostQuantized > 0 ? bestConsensusTranslation.at<double>(0, 0) : translation_mat.at<double>(0, 0);
        ret.ty = mostQuantized > 0 ? bestConsensusTranslation.at<double>(0, 1) : translation_mat.at<double>(0, 1);
        ret.tz = mostQuantized > 0 ? bestConsensusTranslation.at<double>(0, 2) : translation_mat.at<double>(0, 2);
        ret.is_valid = numInliers >= 6;
        delete[] indices;
        return ret;
    } else {
        if (matches.size() < 10) {
            ret.is_valid = false;
            ret.yaw = 0;
            return ret;
        }
        ret.is_valid = true;
        ret.numMatches = vectors1.size();
        const auto yaw = getYaw(vectors1, vectors2, intrinsics1_matrix, ret.numInliers, ret.residualAngle, ret.tx, ret.ty, ret.tz);

        ret.yaw = yaw;
        std::cout << "ret.yaw " << ret.yaw << std::endl;
        return ret;
    }
}
