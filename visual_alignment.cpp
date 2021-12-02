//
//  visual_alignment.cpp
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

int main(int argc, char** argv) {
    VisualFeatureType vftype;
    std::string alignImagePath, alignMetadataPath;
    std::string cameraImagePath, cameraMetadataPath;
    std::string outputDirectory;
    double ratioTest;
    bool debug;

    try {
        po::options_description desc("Allowed options");

        desc.add_options()
            ("help", "describe arguments")
            ("align-image-path", po::value<std::string>(&alignImagePath)->required(), "path to the alignment image file")
            ("align-metadata-path", po::value<std::string>(&alignMetadataPath)->required(), "path to the alignment metadata")
            ("camera-image-path", po::value<std::string>(&cameraImagePath)->required(), "path to the camera image file")
            ("camera-metadata-path", po::value<std::string>(&cameraMetadataPath)->required(), "path to the camera metadata")
            ("output-directory", po::value<std::string>(&outputDirectory)->required(), "path to a directory where outputs will be placed")
            ("vftype", po::value<int>()->required(), "visual feature type\n0: SIFT\n1: ORB\n2: AKAZE MLDB Upright\n3: AKAZE MLDB\n4: SIFT Upright")
            ("ratio-test", po::value<double>(&ratioTest)->default_value(0.7), "ratio test")
            ("debug", po::value<bool>(&debug)->default_value(false), "true if debug windows should be rendered, false (default) otherwise");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 1;
        }
        po::notify(vm);
        if (vm.count("vftype")) {
            switch (vm["vftype"].as<int>()) {
                case 0:
                    vftype = VisualFeatureType::SIFT;
                    break;
                case 1:
                    vftype = VisualFeatureType::ORB;
                    break;
                case 2:
                    vftype = VisualFeatureType::AKAZE_MLDB_UPRIGHT;
                    break;
                case 3:
                    vftype = VisualFeatureType::AKAZE_MLDB;
                    break;
                case 4:
                    vftype = VisualFeatureType::SIFT_UPRIGHT;
                    break;
                default:
                    std::cerr << "invalid vftype " << vm["vftype"].as<int>() << std::endl;
                    return 1;
            }
        }
        struct stat sb;
        if (stat(outputDirectory.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
            std::cerr << "Specified output directory " << outputDirectory << " does not exist or is not a directory" << std::endl;
            return 1;
        }
    } catch (const po::error &ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    // a downsample factor of 2 is good for 1920x1440 images
    const int downSampleFactor = 2;
    json jAlign, jCamera;

    std::ifstream alignMetadata(alignMetadataPath);
    alignMetadata >> jAlign;

    // TODO: more elegance!
    cv::Mat alignIntrinsics  = (cv::Mat_<float>(3,3) << jAlign["intrinsics"][0],
                                                        jAlign["intrinsics"][1],
                                                        jAlign["intrinsics"][2],
                                                        jAlign["intrinsics"][3],
                                                        jAlign["intrinsics"][4],
                                                        jAlign["intrinsics"][5],
                                                        jAlign["intrinsics"][6],
                                                        jAlign["intrinsics"][7],
                                                        jAlign["intrinsics"][8]);

    cv::Mat alignPose  = (cv::Mat_<float>(4,4) << jAlign["transform"][0],
                                                  jAlign["transform"][1],
                                                  jAlign["transform"][2],
                                                  jAlign["transform"][3],
                                                  jAlign["transform"][4],
                                                  jAlign["transform"][5],
                                                  jAlign["transform"][6],
                                                  jAlign["transform"][7],
                                                  jAlign["transform"][8],
                                                  jAlign["transform"][9],
                                                  jAlign["transform"][10],
                                                  jAlign["transform"][11],
                                                  jAlign["transform"][12],
                                                  jAlign["transform"][13],
                                                  jAlign["transform"][14],
                                                  jAlign["transform"][15]);

    cv::Mat alignImage = cv::imread(alignImagePath);

    std::ifstream cameraMetadata(cameraMetadataPath);
    cameraMetadata >> jCamera;

    // TODO: more elegance!
    cv::Mat cameraIntrinsics  = (cv::Mat_<float>(3,3) << jCamera["intrinsics"][0],
                                                         jCamera["intrinsics"][1],
                                                         jCamera["intrinsics"][2],
                                                         jCamera["intrinsics"][3],
                                                         jCamera["intrinsics"][4],
                                                         jCamera["intrinsics"][5],
                                                         jCamera["intrinsics"][6],
                                                         jCamera["intrinsics"][7],
                                                         jCamera["intrinsics"][8]);

    cv::Mat cameraPose  = (cv::Mat_<float>(4,4) << jCamera["transform"][0],
                                                   jCamera["transform"][1],
                                                   jCamera["transform"][2],
                                                   jCamera["transform"][3],
                                                   jCamera["transform"][4],
                                                   jCamera["transform"][5],
                                                   jCamera["transform"][6],
                                                   jCamera["transform"][7],
                                                   jCamera["transform"][8],
                                                   jCamera["transform"][9],
                                                   jCamera["transform"][10],
                                                   jCamera["transform"][11],
                                                   jCamera["transform"][12],
                                                   jCamera["transform"][13],
                                                   jCamera["transform"][14],
                                                   jCamera["transform"][15]);

    cv::Mat cameraImage = cv::imread(cameraImagePath);


    VisualAlignmentReturn ret =  visualYaw(&alignImage, &alignIntrinsics, &alignPose, &cameraImage, &cameraIntrinsics, &cameraPose, 2, vftype, ratioTest, debug);

    if (!ret.matchingImage.empty()) {
        boost::filesystem::path matchingImagePath(outputDirectory);
        matchingImagePath /= "matching_image.png";
        cv::imwrite(matchingImagePath.string(), ret.matchingImage);

        json outputInfo;
        outputInfo["yaw"] = ret.yaw;
        outputInfo["is_valid"] = ret.is_valid;
        outputInfo["numInliers"] = ret.numInliers;
        outputInfo["numMatches"] = ret.numMatches;
        outputInfo["residualAngle"] = ret.residualAngle;
        outputInfo["tx"] = ret.tx;
        outputInfo["ty"] = ret.ty;
        outputInfo["tz"] = ret.tz;
        outputInfo["triangulatedPoints"] = ret.triangulatedPoints;
        outputInfo["image1MatchCoordinates"] = ret.image1MatchCoordinates;
        outputInfo["image2MatchCoordinates"] = ret.image2MatchCoordinates;
        outputInfo["cameraIntrinsicsRowMajor"] = jCamera["intrinsics"];
        outputInfo["cameraPoseRowMajor"] = jCamera["transform"];
        outputInfo["alignIntrinsicsRowMajor"] = jAlign["intrinsics"];
        outputInfo["alignPoseRowMajor"] = jAlign["transform"];
        boost::filesystem::path outputInfoPath(outputDirectory);
        outputInfoPath /= "results.json";
        std::ofstream out(outputInfoPath.string());
        out << outputInfo.dump(4) << std::endl;
    }

    std::cout << "n matches " << ret.numMatches << " n inliers " << ret.numInliers << std::endl;
    return 0;
}
