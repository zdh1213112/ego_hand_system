#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "ego_hand/calibration.hpp"
#include "ego_hand/session.hpp"

namespace {

struct Options {
    std::filesystem::path session;
    std::filesystem::path output_csv;
    std::uint64_t max_delta_us = 1500;
    std::size_t stride = 5;
    std::size_t min_common_markers = 4;
};

struct FrameResult {
    std::size_t pair_index = 0;
    std::size_t left_index = 0;
    std::size_t right_index = 0;
    std::int64_t timestamp_delta_us = 0;
    std::size_t left_markers = 0;
    std::size_t right_markers = 0;
    std::size_t common_markers = 0;
    std::vector<double> vertical_errors_px;
};

void printUsage(const char *program) {
    std::cerr << "Usage: " << program
              << " --session <recording-directory> [--output-csv <file>]"
                 " [--stride 5] [--min-common-markers 4] [--max-delta-us 1500]\n";
}

Options parseOptions(int argc, char **argv) {
    Options options;
    for(int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto next = [&]() -> std::string {
            if(i + 1 >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            return argv[++i];
        };
        if(argument == "--session") {
            options.session = next();
        }
        else if(argument == "--output-csv") {
            options.output_csv = next();
        }
        else if(argument == "--stride") {
            options.stride = std::stoull(next());
        }
        else if(argument == "--min-common-markers") {
            options.min_common_markers = std::stoull(next());
        }
        else if(argument == "--max-delta-us") {
            options.max_delta_us = std::stoull(next());
        }
        else if(argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        }
        else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if(options.session.empty()) {
        throw std::runtime_error("--session is required");
    }
    if(options.stride == 0) {
        throw std::runtime_error("--stride must be greater than zero");
    }
    return options;
}

cv::Mat readFrameAt(cv::VideoCapture &capture, std::size_t target_index, std::size_t &next_index) {
    cv::Mat frame;
    while(next_index <= target_index) {
        if(!capture.read(frame) || frame.empty()) {
            throw std::runtime_error("video ended before requested frame " + std::to_string(target_index));
        }
        ++next_index;
    }
    return frame;
}

std::map<int, std::vector<cv::Point2f>> detectMarkers(
    const cv::Mat &image,
    const cv::Ptr<cv::aruco::Dictionary> &dictionary) {
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    cv::aruco::detectMarkers(image, dictionary, corners, ids);
    std::map<int, std::vector<cv::Point2f>> result;
    for(std::size_t i = 0; i < ids.size(); ++i) {
        result.emplace(ids[i], std::move(corners[i]));
    }
    return result;
}

double percentile(std::vector<double> values, double ratio) {
    if(values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto position = std::ceil(ratio * static_cast<double>(values.size()));
    const auto index = static_cast<std::size_t>(std::max(1.0, position)) - 1;
    return values[std::min(index, values.size() - 1)];
}

void writeCsv(const std::filesystem::path &path, const std::vector<FrameResult> &results) {
    if(path.empty()) {
        return;
    }
    if(!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if(!output) {
        throw std::runtime_error("cannot create CSV: " + path.string());
    }
    output << "pair_index,left_index,right_index,timestamp_delta_us,left_markers,right_markers,"
              "common_markers,corner_count,vertical_abs_median_px,vertical_abs_p95_px,vertical_abs_max_px\n";
    for(const auto &result: results) {
        std::vector<double> absolute_errors;
        absolute_errors.reserve(result.vertical_errors_px.size());
        for(double value: result.vertical_errors_px) {
            absolute_errors.push_back(std::abs(value));
        }
        output << result.pair_index << ',' << result.left_index << ',' << result.right_index << ','
               << result.timestamp_delta_us << ',' << result.left_markers << ',' << result.right_markers << ','
               << result.common_markers << ',' << absolute_errors.size() << ','
               << percentile(absolute_errors, 0.50) << ',' << percentile(absolute_errors, 0.95) << ','
               << percentile(absolute_errors, 1.00) << '\n';
    }
}

}  // namespace

int main(int argc, char **argv) try {
    const Options options = parseOptions(argc, argv);
    const auto session = ego_hand::discoverSession(options.session);
    const auto calibration = ego_hand::loadStereoCalibration(session.camera_calibration);
    const auto validation = ego_hand::validateCalibration(calibration);
    if(!validation.ok) {
        throw std::runtime_error("camera calibration is invalid");
    }
    const auto rectification = ego_hand::createFisheyeRectification(calibration);
    const auto left_timestamps = ego_hand::loadTimestampCsv(session.left_timestamps);
    const auto right_timestamps = ego_hand::loadTimestampCsv(session.right_timestamps);
    const auto pairs = ego_hand::pairTimestamps(left_timestamps, right_timestamps, options.max_delta_us);
    if(pairs.empty()) {
        throw std::runtime_error("no synchronized stereo pairs found");
    }

    cv::VideoCapture left_capture(session.left_video.string());
    cv::VideoCapture right_capture(session.right_video.string());
    if(!left_capture.isOpened() || !right_capture.isOpened()) {
        throw std::runtime_error("failed to open one or both videos");
    }

    // The recorded validation target uses IDs 0..49 from the predefined 5x5 dictionary.
    const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
    std::vector<FrameResult> results;
    std::vector<double> all_vertical_errors;
    std::size_t left_next = 0;
    std::size_t right_next = 0;
    std::size_t sampled_frames = 0;

    for(std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
        const auto &pair = pairs[pair_index];
        cv::Mat left = readFrameAt(left_capture, pair.left_index, left_next);
        cv::Mat right = readFrameAt(right_capture, pair.right_index, right_next);
        if(pair_index % options.stride != 0) {
            continue;
        }
        ++sampled_frames;

        cv::Mat left_rectified;
        cv::Mat right_rectified;
        cv::remap(left, left_rectified, rectification.map_left_x, rectification.map_left_y, cv::INTER_LINEAR);
        cv::remap(right, right_rectified, rectification.map_right_x, rectification.map_right_y, cv::INTER_LINEAR);
        auto left_markers = detectMarkers(left_rectified, dictionary);
        auto right_markers = detectMarkers(right_rectified, dictionary);

        FrameResult frame_result;
        frame_result.pair_index = pair_index;
        frame_result.left_index = pair.left_index;
        frame_result.right_index = pair.right_index;
        frame_result.timestamp_delta_us = pair.deltaUs();
        frame_result.left_markers = left_markers.size();
        frame_result.right_markers = right_markers.size();

        for(const auto &[id, left_corners]: left_markers) {
            const auto right_it = right_markers.find(id);
            if(right_it == right_markers.end() || left_corners.size() != right_it->second.size()) {
                continue;
            }
            ++frame_result.common_markers;
            for(std::size_t corner = 0; corner < left_corners.size(); ++corner) {
                frame_result.vertical_errors_px.push_back(right_it->second[corner].y - left_corners[corner].y);
            }
        }

        if(frame_result.common_markers >= options.min_common_markers) {
            all_vertical_errors.insert(all_vertical_errors.end(),
                                       frame_result.vertical_errors_px.begin(),
                                       frame_result.vertical_errors_px.end());
            results.push_back(std::move(frame_result));
        }
    }

    writeCsv(options.output_csv, results);
    if(all_vertical_errors.empty()) {
        std::cout << "Sampled stereo pairs: " << sampled_frames << "\n"
                  << "Usable board frames: 0\n"
                  << "No frame contained at least " << options.min_common_markers
                  << " common DICT_5X5_50 markers.\n";
        return 2;
    }

    std::vector<double> absolute_errors;
    absolute_errors.reserve(all_vertical_errors.size());
    double signed_sum = 0.0;
    for(double value: all_vertical_errors) {
        signed_sum += value;
        absolute_errors.push_back(std::abs(value));
    }

    std::cout << std::fixed << std::setprecision(3)
              << "EGO epipolar validation\n"
              << "Session: " << session.root << "\n"
              << "Dictionary: DICT_5X5_50\n"
              << "Sampled stereo pairs: " << sampled_frames << " (stride=" << options.stride << ")\n"
              << "Usable board frames: " << results.size() << "\n"
              << "Matched marker corners: " << absolute_errors.size() << "\n"
              << "Signed vertical mean: " << signed_sum / static_cast<double>(all_vertical_errors.size()) << " px\n"
              << "Absolute vertical median/P95/max: "
              << percentile(absolute_errors, 0.50) << "/"
              << percentile(absolute_errors, 0.95) << "/"
              << percentile(absolute_errors, 1.00) << " px\n";
    if(!options.output_csv.empty()) {
        std::cout << "Per-frame CSV: " << options.output_csv << "\n";
    }
    return 0;
}
catch(const std::exception &error) {
    std::cerr << "ego_epipolar_validate: " << error.what() << "\n";
    printUsage(argv[0]);
    return 1;
}

