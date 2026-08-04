#include "ego_hand/calibration.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>

namespace ego_hand {
namespace {

double requiredDouble(const YAML::Node &node, const char *key) {
    if(!node[key]) {
        throw std::runtime_error(std::string("missing calibration field: ") + key);
    }
    return node[key].as<double>();
}

cv::Matx33d parseRotation(const YAML::Node &node) {
    if(!node || node.size() != 3) {
        throw std::runtime_error("rotation must contain three rows");
    }
    cv::Matx33d result;
    for(int row = 0; row < 3; ++row) {
        if(node[row].size() != 3) {
            throw std::runtime_error("rotation row must contain three values");
        }
        for(int col = 0; col < 3; ++col) {
            result(row, col) = node[row][col].as<double>();
        }
    }
    return result;
}

CameraCalibration parseCamera(const YAML::Node &node) {
    CameraCalibration camera;
    camera.id = node["id"].as<std::string>();
    camera.name = node["name"].as<std::string>();
    camera.distortion_model = node["distortion_model"].as<std::string>();
    camera.image_size = cv::Size(node["image_width"].as<int>(), node["image_height"].as<int>());

    const auto intrinsics = node["intrinsics"];
    const double fx = requiredDouble(intrinsics, "fx");
    const double fy = requiredDouble(intrinsics, "fy");
    const double cx = requiredDouble(intrinsics, "cx");
    const double cy = requiredDouble(intrinsics, "cy");
    camera.K = cv::Matx33d(fx, 0.0, cx,
                           0.0, fy, cy,
                           0.0, 0.0, 1.0);

    const auto distortion = node["distortion"];
    camera.D = cv::Vec4d(requiredDouble(distortion, "k1"),
                         requiredDouble(distortion, "k2"),
                         requiredDouble(distortion, "k3"),
                         requiredDouble(distortion, "k4"));

    const auto extrinsics = node["extrinsics"];
    camera.R_reference_to_camera = parseRotation(extrinsics["rotation"]);
    const auto translation = extrinsics["translation"];
    if(!translation || translation.size() != 3) {
        throw std::runtime_error("translation must contain three values");
    }
    // The EGO calibration file stores translation in millimetres. Internally we use metres.
    camera.t_reference_to_camera_m = cv::Vec3d(translation[0].as<double>(),
                                               translation[1].as<double>(),
                                               translation[2].as<double>()) * 1e-3;
    return camera;
}

double rotationOrthogonalityError(const cv::Matx33d &R) {
    const cv::Matx33d residual = R * R.t() - cv::Matx33d::eye();
    return cv::norm(cv::Mat(residual), cv::NORM_INF);
}

}  // namespace

double StereoCalibration::baselineMeters() const {
    return cv::norm(right.t_reference_to_camera_m - left.t_reference_to_camera_m);
}

StereoCalibration loadStereoCalibration(const std::filesystem::path &yaml_path) {
    const YAML::Node root = YAML::LoadFile(yaml_path.string());
    const YAML::Node info = root["calibration_info"];
    const YAML::Node cameras = root["cameras"];
    if(!info || !cameras || cameras.size() != 2) {
        throw std::runtime_error("expected calibration_info and exactly two cameras");
    }

    StereoCalibration calibration;
    calibration.calibration_serial = info["serial_number"].as<std::string>();
    calibration.reference_camera = info["reference_camera"].as<std::string>();
    calibration.left = parseCamera(cameras[0]);
    calibration.right = parseCamera(cameras[1]);
    return calibration;
}

CalibrationValidation validateCalibration(const StereoCalibration &calibration) {
    CalibrationValidation result;
    const auto validate_camera = [&](const CameraCalibration &camera, const char *side) {
        if(camera.image_size.width <= 0 || camera.image_size.height <= 0) {
            result.errors.emplace_back(std::string(side) + " camera has invalid image size");
        }
        if(camera.K(0, 0) <= 0.0 || camera.K(1, 1) <= 0.0) {
            result.errors.emplace_back(std::string(side) + " camera has non-positive focal length");
        }
        if(camera.distortion_model != "KB") {
            result.errors.emplace_back(std::string(side) + " camera distortion is not KB; current rectifier only supports KB");
        }
        if(rotationOrthogonalityError(camera.R_reference_to_camera) > 1e-4 ||
           std::abs(cv::determinant(cv::Mat(camera.R_reference_to_camera)) - 1.0) > 1e-4) {
            result.errors.emplace_back(std::string(side) + " camera rotation is not a valid rotation matrix");
        }
    };

    validate_camera(calibration.left, "left");
    validate_camera(calibration.right, "right");
    if(calibration.left.image_size != calibration.right.image_size) {
        result.errors.emplace_back("left/right calibration resolutions differ");
    }
    const double baseline = calibration.baselineMeters();
    if(baseline < 0.02 || baseline > 0.50) {
        std::ostringstream message;
        message << "unexpected stereo baseline: " << baseline << " m";
        result.errors.push_back(message.str());
    }
    if(calibration.calibration_serial.empty()) {
        result.warnings.emplace_back("calibration serial number is empty");
    }
    result.ok = result.errors.empty();
    return result;
}

Rectification createFisheyeRectification(const StereoCalibration &calibration, double balance) {
    const auto validation = validateCalibration(calibration);
    if(!validation.ok) {
        throw std::runtime_error("cannot rectify invalid calibration: " + validation.errors.front());
    }

    // Extrinsics in the YAML transform a point from cam_0/reference to cam_1.
    const cv::Matx33d R_left_to_right = calibration.right.R_reference_to_camera *
                                        calibration.left.R_reference_to_camera.t();
    const cv::Vec3d t_left_to_right = calibration.right.t_reference_to_camera_m -
                                      R_left_to_right * calibration.left.t_reference_to_camera_m;

    Rectification result;
    cv::fisheye::stereoRectify(calibration.left.K,
                               calibration.left.D,
                               calibration.right.K,
                               calibration.right.D,
                               calibration.left.image_size,
                               R_left_to_right,
                               t_left_to_right,
                               result.R1,
                               result.R2,
                               result.P1,
                               result.P2,
                               result.Q,
                               cv::CALIB_ZERO_DISPARITY,
                               calibration.left.image_size,
                               balance,
                               1.0);

    cv::fisheye::initUndistortRectifyMap(calibration.left.K,
                                         calibration.left.D,
                                         result.R1,
                                         result.P1,
                                         calibration.left.image_size,
                                         CV_32FC1,
                                         result.map_left_x,
                                         result.map_left_y);
    cv::fisheye::initUndistortRectifyMap(calibration.right.K,
                                         calibration.right.D,
                                         result.R2,
                                         result.P2,
                                         calibration.right.image_size,
                                         CV_32FC1,
                                         result.map_right_x,
                                         result.map_right_y);
    return result;
}

}  // namespace ego_hand

