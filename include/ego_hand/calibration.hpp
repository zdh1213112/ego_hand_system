#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace ego_hand {

struct CameraCalibration {
    std::string id;
    std::string name;
    std::string distortion_model;
    cv::Size image_size;
    cv::Matx33d K = cv::Matx33d::eye();
    cv::Vec4d D = cv::Vec4d::all(0.0);
    cv::Matx33d R_reference_to_camera = cv::Matx33d::eye();
    cv::Vec3d t_reference_to_camera_m = cv::Vec3d::all(0.0);
};

struct StereoCalibration {
    std::string calibration_serial;
    std::string reference_camera;
    CameraCalibration left;
    CameraCalibration right;

    double baselineMeters() const;
};

struct CalibrationValidation {
    bool ok = false;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

struct Rectification {
    cv::Matx33d R1 = cv::Matx33d::eye();
    cv::Matx33d R2 = cv::Matx33d::eye();
    cv::Matx34d P1 = cv::Matx34d::zeros();
    cv::Matx34d P2 = cv::Matx34d::zeros();
    cv::Matx44d Q = cv::Matx44d::zeros();
    cv::Mat map_left_x;
    cv::Mat map_left_y;
    cv::Mat map_right_x;
    cv::Mat map_right_y;
};

StereoCalibration loadStereoCalibration(const std::filesystem::path &yaml_path);
CalibrationValidation validateCalibration(const StereoCalibration &calibration);
Rectification createFisheyeRectification(const StereoCalibration &calibration, double balance = 0.0);

}  // namespace ego_hand

