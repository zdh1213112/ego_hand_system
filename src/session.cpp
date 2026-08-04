#include "ego_hand/session.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ego_hand {
namespace {

std::filesystem::path findUniqueSuffix(const std::filesystem::path &root,
                                       const std::string &suffix,
                                       bool required = true) {
    std::vector<std::filesystem::path> matches;
    for(const auto &entry: std::filesystem::directory_iterator(root)) {
        if(!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if(name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            matches.push_back(entry.path());
        }
    }
    if(matches.size() > 1) {
        throw std::runtime_error("multiple files end with " + suffix + " in " + root.string());
    }
    if(matches.empty()) {
        if(required) {
            throw std::runtime_error("missing file ending with " + suffix + " in " + root.string());
        }
        return {};
    }
    return matches.front();
}

std::uint64_t percentile(std::vector<std::uint64_t> values, double ratio) {
    if(values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(std::ceil(ratio * static_cast<double>(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
}

}  // namespace

std::int64_t TimestampPair::deltaUs() const {
    return static_cast<std::int64_t>(right_timestamp_us) - static_cast<std::int64_t>(left_timestamp_us);
}

EgoSession discoverSession(const std::filesystem::path &root) {
    if(!std::filesystem::is_directory(root)) {
        throw std::runtime_error("session directory does not exist: " + root.string());
    }
    EgoSession session;
    session.root = std::filesystem::canonical(root);
    session.camera_calibration = findUniqueSuffix(session.root, "_calibration_camera.yaml");
    session.imu_calibration = findUniqueSuffix(session.root, "_calibration_imu.yaml");
    session.left_video = findUniqueSuffix(session.root, "_camera_left.mp4");
    session.right_video = findUniqueSuffix(session.root, "_camera_right.mp4");
    session.left_timestamps = findUniqueSuffix(session.root, "_camera_left_pts.csv");
    session.right_timestamps = findUniqueSuffix(session.root, "_camera_right_pts.csv");
    session.imu_csv = findUniqueSuffix(session.root, "_imu.csv");
    session.metadata_json = findUniqueSuffix(session.root, ".json");
    return session;
}

std::vector<std::uint64_t> loadTimestampCsv(const std::filesystem::path &csv_path) {
    std::ifstream input(csv_path);
    if(!input) {
        throw std::runtime_error("cannot open timestamp file: " + csv_path.string());
    }

    std::string line;
    if(!std::getline(input, line) || line.find("timestamp_us") == std::string::npos) {
        throw std::runtime_error("invalid timestamp CSV header: " + csv_path.string());
    }

    std::vector<std::uint64_t> timestamps;
    while(std::getline(input, line)) {
        if(line.empty()) {
            continue;
        }
        const auto comma = line.find(',');
        timestamps.push_back(std::stoull(line.substr(0, comma)));
    }
    if(timestamps.empty()) {
        throw std::runtime_error("timestamp CSV is empty: " + csv_path.string());
    }
    for(std::size_t i = 1; i < timestamps.size(); ++i) {
        if(timestamps[i] <= timestamps[i - 1]) {
            throw std::runtime_error("timestamps are not strictly increasing: " + csv_path.string());
        }
    }
    return timestamps;
}

std::vector<TimestampPair> pairTimestamps(const std::vector<std::uint64_t> &left,
                                          const std::vector<std::uint64_t> &right,
                                          std::uint64_t max_delta_us) {
    std::vector<TimestampPair> result;
    std::size_t i = 0;
    std::size_t j = 0;
    while(i < left.size() && j < right.size()) {
        const std::int64_t delta = static_cast<std::int64_t>(right[j]) - static_cast<std::int64_t>(left[i]);
        const auto absolute_delta = static_cast<std::uint64_t>(std::llabs(delta));
        if(absolute_delta <= max_delta_us) {
            result.push_back({i, j, left[i], right[j]});
            ++i;
            ++j;
        }
        else if(delta > 0) {
            ++i;
        }
        else {
            ++j;
        }
    }
    return result;
}

PairingStatistics calculatePairingStatistics(const std::vector<std::uint64_t> &left,
                                              const std::vector<std::uint64_t> &right,
                                              const std::vector<TimestampPair> &pairs) {
    PairingStatistics result;
    result.left_count = left.size();
    result.right_count = right.size();
    result.pair_count = pairs.size();
    result.skipped_left = left.size() - pairs.size();
    result.skipped_right = right.size() - pairs.size();

    std::vector<std::uint64_t> absolute_deltas;
    absolute_deltas.reserve(pairs.size());
    for(const auto &pair: pairs) {
        absolute_deltas.push_back(static_cast<std::uint64_t>(std::llabs(pair.deltaUs())));
    }
    result.median_abs_delta_us = percentile(absolute_deltas, 0.50);
    result.p95_abs_delta_us = percentile(absolute_deltas, 0.95);
    result.max_abs_delta_us = percentile(absolute_deltas, 1.00);
    return result;
}

}  // namespace ego_hand
