#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "ego_hand/calibration.hpp"
#include "ego_hand/session.hpp"

namespace {

struct Options {
    std::filesystem::path session;
    std::filesystem::path output;
    std::uint64_t max_delta_us = 1500;
    std::size_t sample_pair = 100;
};

void printUsage(const char *program) {
    std::cerr << "Usage: " << program
              << " --session <recording-directory> [--output <directory>]"
                 " [--max-delta-us 1500] [--sample-pair 100]\n";
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
        else if(argument == "--output") {
            options.output = next();
        }
        else if(argument == "--max-delta-us") {
            options.max_delta_us = std::stoull(next());
        }
        else if(argument == "--sample-pair") {
            options.sample_pair = std::stoull(next());
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
    return options;
}

cv::Mat readFrameAt(cv::VideoCapture &capture, std::size_t target_index, std::size_t &next_index) {
    cv::Mat frame;
    while(next_index <= target_index) {
        if(!capture.read(frame) || frame.empty()) {
            throw std::runtime_error("video ended before requested frame index " + std::to_string(target_index));
        }
        ++next_index;
    }
    return frame;
}

cv::Mat makePairImage(const cv::Mat &left, const cv::Mat &right, const std::string &title) {
    cv::Mat pair;
    cv::hconcat(left, right, pair);
    for(int y = 100; y < pair.rows; y += 100) {
        cv::line(pair, cv::Point(0, y), cv::Point(pair.cols - 1, y), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    cv::putText(pair, title, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    cv::putText(pair, "LEFT", cv::Point(30, pair.rows - 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    cv::putText(pair, "RIGHT", cv::Point(left.cols + 30, pair.rows - 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    return pair;
}

}  // namespace

int main(int argc, char **argv) try {
    const Options options = parseOptions(argc, argv);
    const auto session = ego_hand::discoverSession(options.session);
    const auto calibration = ego_hand::loadStereoCalibration(session.camera_calibration);
    const auto validation = ego_hand::validateCalibration(calibration);
    const auto left_timestamps = ego_hand::loadTimestampCsv(session.left_timestamps);
    const auto right_timestamps = ego_hand::loadTimestampCsv(session.right_timestamps);
    const auto pairs = ego_hand::pairTimestamps(left_timestamps, right_timestamps, options.max_delta_us);
    const auto statistics = ego_hand::calculatePairingStatistics(left_timestamps, right_timestamps, pairs);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "EGO session: " << session.root << "\n"
              << "Calibration serial: " << calibration.calibration_serial << "\n"
              << "Calibration resolution: " << calibration.left.image_size.width << "x"
              << calibration.left.image_size.height << "\n"
              << "Distortion model: " << calibration.left.distortion_model << "\n"
              << "Left intrinsics: fx=" << calibration.left.K(0, 0)
              << " fy=" << calibration.left.K(1, 1)
              << " cx=" << calibration.left.K(0, 2)
              << " cy=" << calibration.left.K(1, 2) << "\n"
              << "Right intrinsics: fx=" << calibration.right.K(0, 0)
              << " fy=" << calibration.right.K(1, 1)
              << " cx=" << calibration.right.K(0, 2)
              << " cy=" << calibration.right.K(1, 2) << "\n"
              << "Stereo baseline: " << calibration.baselineMeters() * 1000.0 << " mm\n"
              << "Calibration valid: " << (validation.ok ? "yes" : "no") << "\n";

    for(const auto &warning: validation.warnings) {
        std::cout << "WARNING: " << warning << "\n";
    }
    for(const auto &error: validation.errors) {
        std::cout << "ERROR: " << error << "\n";
    }

    std::cout << "Left timestamp count: " << statistics.left_count << "\n"
              << "Right timestamp count: " << statistics.right_count << "\n"
              << "Paired frames: " << statistics.pair_count << "\n"
              << "Unpaired left/right: " << statistics.skipped_left << "/" << statistics.skipped_right << "\n"
              << "Absolute sync delta median/P95/max: "
              << statistics.median_abs_delta_us << "/"
              << statistics.p95_abs_delta_us << "/"
              << statistics.max_abs_delta_us << " us\n";

    if(!validation.ok) {
        return 2;
    }
    if(pairs.empty()) {
        throw std::runtime_error("no stereo pairs found within the requested threshold");
    }

    if(!options.output.empty()) {
        std::filesystem::create_directories(options.output);
        const std::size_t pair_index = std::min(options.sample_pair, pairs.size() - 1);
        const auto &sample = pairs[pair_index];

        cv::VideoCapture left_capture(session.left_video.string());
        cv::VideoCapture right_capture(session.right_video.string());
        if(!left_capture.isOpened() || !right_capture.isOpened()) {
            throw std::runtime_error("failed to open one or both MP4 files");
        }

        std::size_t left_next = 0;
        std::size_t right_next = 0;
        const cv::Mat left = readFrameAt(left_capture, sample.left_index, left_next);
        const cv::Mat right = readFrameAt(right_capture, sample.right_index, right_next);
        if(left.size() != calibration.left.image_size || right.size() != calibration.right.image_size) {
            throw std::runtime_error("video resolution does not match calibration resolution");
        }

        const auto rectification = ego_hand::createFisheyeRectification(calibration);
        cv::Mat left_rectified;
        cv::Mat right_rectified;
        cv::remap(left, left_rectified, rectification.map_left_x, rectification.map_left_y, cv::INTER_LINEAR);
        cv::remap(right, right_rectified, rectification.map_right_x, rectification.map_right_y, cv::INTER_LINEAR);

        const auto raw_path = options.output / "stereo_raw.jpg";
        const auto rectified_path = options.output / "stereo_rectified.jpg";
        const std::string timing = "pair=" + std::to_string(pair_index) +
                                   " dt=" + std::to_string(sample.deltaUs()) + " us";
        if(!cv::imwrite(raw_path.string(), makePairImage(left, right, "RAW " + timing)) ||
           !cv::imwrite(rectified_path.string(), makePairImage(left_rectified, right_rectified, "RECTIFIED " + timing))) {
            throw std::runtime_error("failed to write preview images");
        }
        std::cout << "Preview raw: " << raw_path << "\n"
                  << "Preview rectified: " << rectified_path << "\n"
                  << "Sample indices left/right: " << sample.left_index << "/" << sample.right_index << "\n";
    }

    return 0;
}
catch(const std::exception &error) {
    std::cerr << "ego_session_inspect: " << error.what() << "\n";
    printUsage(argv[0]);
    return 1;
}

