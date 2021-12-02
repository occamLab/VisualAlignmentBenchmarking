//
//  sfm.cpp
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
    std::string image1Path, metadata1Path, image2Path, metadata2Path;
    std::string outputDirectory;
    double ratioTest;
    bool debug;

    try {
        po::options_description desc("Allowed options");

        desc.add_options()
            ("help", "describe arguments")
            ("image-1-path", po::value<std::string>(&image1Path)->required(), "path to the first image file")
            ("metadata-1-path", po::value<std::string>(&metadata1Path)->required(), "path to the first image's metadata")
            ("image-2-path", po::value<std::string>(&image2Path)->required(), "path to the second image file")
            ("metadata-2-path", po::value<std::string>(&metadata2Path)->required(), "path to the second image's metadata")
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
    json jImage1, jImage2;

    std::ifstream metadata1(metadata1Path);
    metadata1 >> jImage1;

    // TODO: more elegance!
    cv::Mat image1Intrinsics  = (cv::Mat_<float>(3,3) << jImage1["intrinsics"][0],
                                                         jImage1["intrinsics"][1],
                                                         jImage1["intrinsics"][2],
                                                         jImage1["intrinsics"][3],
                                                         jImage1["intrinsics"][4],
                                                         jImage1["intrinsics"][5],
                                                         jImage1["intrinsics"][6],
                                                         jImage1["intrinsics"][7],
                                                         jImage1["intrinsics"][8]);

    cv::Mat pose1  = (cv::Mat_<float>(4,4) << jImage1["transform"][0],
                                              jImage1["transform"][1],
                                              jImage1["transform"][2],
                                              jImage1["transform"][3],
                                              jImage1["transform"][4],
                                              jImage1["transform"][5],
                                              jImage1["transform"][6],
                                              jImage1["transform"][7],
                                              jImage1["transform"][8],
                                              jImage1["transform"][9],
                                              jImage1["transform"][10],
                                              jImage1["transform"][11],
                                              jImage1["transform"][12],
                                              jImage1["transform"][13],
                                              jImage1["transform"][14],
                                              jImage1["transform"][15]);

    cv::Mat image1 = cv::imread(image1Path);

    std::ifstream metadata2(metadata2Path);
    metadata2 >> jImage2;

    // TODO: more elegance!
    cv::Mat image2Intrinsics  = (cv::Mat_<float>(3,3) << jImage2["intrinsics"][0],
                                                         jImage2["intrinsics"][1],
                                                         jImage2["intrinsics"][2],
                                                         jImage2["intrinsics"][3],
                                                         jImage2["intrinsics"][4],
                                                         jImage2["intrinsics"][5],
                                                         jImage2["intrinsics"][6],
                                                         jImage2["intrinsics"][7],
                                                         jImage2["intrinsics"][8]);

    cv::Mat pose2  = (cv::Mat_<float>(4,4) << jImage2["transform"][0],
                                              jImage2["transform"][1],
                                              jImage2["transform"][2],
                                              jImage2["transform"][3],
                                              jImage2["transform"][4],
                                              jImage2["transform"][5],
                                              jImage2["transform"][6],
                                              jImage2["transform"][7],
                                              jImage2["transform"][8],
                                              jImage2["transform"][9],
                                              jImage2["transform"][10],
                                              jImage2["transform"][11],
                                              jImage2["transform"][12],
                                              jImage2["transform"][13],
                                              jImage2["transform"][14],
                                              jImage2["transform"][15]);

    cv::Mat image2 = cv::imread(image2Path);


    VisualAlignmentReturn ret =  visualYaw(&image1, &image1Intrinsics, &pose1, &image2, &image2Intrinsics, &pose2, 2, vftype, ratioTest, debug);

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
        outputInfo["intrinsics2RowMajor"] = jImage2["intrinsics"];
        outputInfo["pose2RowMajor"] = jImage2["transform"];
        outputInfo["intrinsics1RowMajor"] = jImage1["intrinsics"];
        outputInfo["pose1RowMajor"] = jImage1["transform"];
        boost::filesystem::path outputInfoPath(outputDirectory);
        outputInfoPath /= "results.json";
        std::ofstream out(outputInfoPath.string());
        out << outputInfo.dump(4) << std::endl;
    }

    std::cout << "n matches " << ret.numMatches << " n inliers " << ret.numInliers << std::endl;
    return 0;
}
