#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ego_hand {

struct EgoSession {
    std::filesystem::path root;
    std::filesystem::path camera_calibration;
    std::filesystem::path imu_calibration;
    std::filesystem::path left_video;
    std::filesystem::path right_video;
    std::filesystem::path left_timestamps;
    std::filesystem::path right_timestamps;
    std::filesystem::path imu_csv;
    std::filesystem::path metadata_json;
};

struct TimestampPair {
    std::size_t left_index = 0;
    std::size_t right_index = 0;
    std::uint64_t left_timestamp_us = 0;
    std::uint64_t right_timestamp_us = 0;

    std::int64_t deltaUs() const;
};

struct PairingStatistics {
    std::size_t left_count = 0;
    std::size_t right_count = 0;
    std::size_t pair_count = 0;
    std::size_t skipped_left = 0;
    std::size_t skipped_right = 0;
    std::uint64_t median_abs_delta_us = 0;
    std::uint64_t p95_abs_delta_us = 0;
    std::uint64_t max_abs_delta_us = 0;
};

EgoSession discoverSession(const std::filesystem::path &root);
std::vector<std::uint64_t> loadTimestampCsv(const std::filesystem::path &csv_path);
std::vector<TimestampPair> pairTimestamps(const std::vector<std::uint64_t> &left,
                                          const std::vector<std::uint64_t> &right,
                                          std::uint64_t max_delta_us);
PairingStatistics calculatePairingStatistics(const std::vector<std::uint64_t> &left,
                                              const std::vector<std::uint64_t> &right,
                                              const std::vector<TimestampPair> &pairs);

}  // namespace ego_hand

