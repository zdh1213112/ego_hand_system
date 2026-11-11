// Continuous stereo/IMU adapter using the pinned Basalt 0.1.7 public queues.
// Basalt itself remains in third_party under its original BSD-3-Clause license.
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <basalt/serialization/headers_serialization.h>
#include <basalt/vi_estimator/vio_estimator.h>
#include <tbb/global_control.h>

namespace {
#pragma pack(push, 1)
struct Header {
    char magic[4];
    uint16_t version;
    uint16_t flags;
    int64_t image_ns;
    uint32_t imu_count;
    uint32_t image_size[2];
    uint32_t reserved;
};
struct Sample {
    int64_t t_ns;
    double gyro[3];
    double accel[3];
};
#pragma pack(pop)
static_assert(sizeof(Header) == 32);
static_assert(sizeof(Sample) == 56);

bool read_header(Header& header) {
    std::cin.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (std::cin.gcount() == 0 && std::cin.eof()) return false;
    if (std::cin.gcount() != sizeof(header)) throw std::runtime_error("truncated Basalt input header");
    if (std::string(header.magic, 4) != "BVIO" || header.version != 2 || header.flags || header.reserved ||
        header.imu_count > 8192 || header.image_ns <= 0 ||
        !header.image_size[0] || !header.image_size[1] ||
        header.image_size[0] > 32'000'000 || header.image_size[1] > 32'000'000)
        throw std::runtime_error("invalid Basalt input header");
    return true;
}

void read_bytes(char* data, size_t size) {
    if (!std::cin.read(data, static_cast<std::streamsize>(size)))
        throw std::runtime_error("truncated Basalt input payload");
}

template<class Matrix>
void write_matrix(std::ostream& stream, const Matrix& matrix) {
    stream << '[';
    for (int row = 0; row < matrix.rows(); ++row) {
        if (row) stream << ',';
        stream << '[';
        for (int column = 0; column < matrix.cols(); ++column) {
            if (column) stream << ',';
            stream << matrix(row, column);
        }
        stream << ']';
    }
    stream << ']';
}

struct Engine {
    basalt::Calibration<double> calibration;
    basalt::VioConfig config;
    basalt::OpticalFlowBase::Ptr flow;
    basalt::VioEstimatorBase::Ptr vio;
    tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> states;
    tbb::concurrent_bounded_queue<basalt::VioVisualizationData::Ptr> maps;
    std::thread state_thread, map_thread;
    std::ofstream output;
    std::mutex output_mutex;
    std::string output_path;
    int64_t shift_ns;
    int64_t last_image_ns = 0, last_imu_ns = 0;
    std::atomic<uint64_t> image_count{0}, imu_count{0}, pose_count{0}, map_count{0};
    bool finished = false;
    bool initialized = false;

    Engine(const std::string& calibration_path, const std::string& config_path,
           const std::string& output_path, int64_t clock_shift)
        : output_path(output_path), shift_ns(clock_shift) {
        std::ifstream input(calibration_path);
        if (!input) throw std::runtime_error("cannot open Basalt calibration");
        cereal::JSONInputArchive archive(input);
        archive(calibration);
        if (calibration.intrinsics.size() != 2 || calibration.T_i_c.size() != 2)
            throw std::runtime_error("live Basalt requires exactly two calibrated cameras");
        config.load(config_path);
        config.vio_enforce_realtime = true;
        std::ifstream existing(output_path, std::ios::binary | std::ios::ate);
        if (existing && existing.tellg() > 0) throw std::runtime_error("VIO output is not empty");
        output.open(output_path, std::ios::out | std::ios::app);
        if (!output) throw std::runtime_error("cannot create VIO output");
        output << std::setprecision(17);
        flow = basalt::OpticalFlowFactory::getOpticalFlow(config, calibration);
        vio = basalt::VioEstimatorFactory::getVioEstimator(config, calibration, basalt::constants::g, true, false);
        flow->input_queue.set_capacity(4);
        vio->imu_data_queue.set_capacity(12000);
        vio->out_state_queue = &states;
        vio->out_vis_queue = &maps;
        flow->output_queue = &vio->vision_data_queue;
        state_thread = std::thread([this] {
            basalt::PoseVelBiasState<double>::Ptr state;
            while (true) {
                states.pop(state);
                if (!state) break;
                const Eigen::Matrix4d pose = (state->T_w_i * calibration.T_i_c[0]).matrix();
                if (!pose.allFinite()) {
                    std::cerr << "Basalt produced a non-finite pose\n";
                    std::_Exit(3);
                }
                std::lock_guard<std::mutex> lock(output_mutex);
                output << "{\"timestamp_us\":{\"cam_0\":" << (state->t_ns-shift_ns)/1000
                       << "},\"vio\":{\"T_world_cam0\":";
                write_matrix(output, pose);
                output << ",\"source\":\"basalt_native_streaming\"}}\n";
                output.flush();
                ++pose_count;
            }
        });
        map_thread = std::thread([this] {
            basalt::VioVisualizationData::Ptr map;
            while (true) {
                maps.pop(map);
                if (!map) break;
                if (map->points.empty()) continue;
                std::lock_guard<std::mutex> lock(output_mutex);
                output << "{\"timestamp_us\":{\"cam_0\":" << (map->t_ns-shift_ns)/1000
                       << "},\"map_points_world_m\":[";
                size_t written = 0;
                const size_t step = std::max<size_t>(1, (map->points.size()+1999)/2000);
                for (size_t index = 0; index < map->points.size(); index += step) {
                    const auto& point = map->points[index];
                    if (!point.allFinite()) continue;
                    if (written++) output << ',';
                    output << '[' << point.x() << ',' << point.y() << ',' << point.z() << ']';
                }
                output << "],\"map_source\":\"basalt_active_sparse_landmarks\"}\n";
                output.flush();
                ++map_count;
            }
        });
        std::cerr << "Basalt continuous estimator ready, cameras=2, scale-res="
                  << calibration.resolution[0].transpose() << '\n';
    }

    void add(const Header& header) {
        if (header.image_ns <= last_image_ns) throw std::runtime_error("image timestamps do not increase");
        std::vector<Sample> samples(header.imu_count);
        read_bytes(reinterpret_cast<char*>(samples.data()), samples.size()*sizeof(Sample));
        Eigen::Vector3d mean_gyro = Eigen::Vector3d::Zero(), mean_accel = Eigen::Vector3d::Zero();
        Eigen::Vector3d bias_gyro = Eigen::Vector3d::Zero(), bias_accel = Eigen::Vector3d::Zero();
        bool stationary_bootstrap = false;
        if (!initialized && samples.size() >= 100) {
            Eigen::Vector3d square_gyro = Eigen::Vector3d::Zero(), square_accel = Eigen::Vector3d::Zero();
            for (const auto& sample : samples) {
                const Eigen::Vector3d gyro(sample.gyro[0], sample.gyro[1], sample.gyro[2]);
                const Eigen::Vector3d accel(sample.accel[0], sample.accel[1], sample.accel[2]);
                mean_gyro += gyro;
                mean_accel += accel;
                square_gyro += gyro.cwiseProduct(gyro);
                square_accel += accel.cwiseProduct(accel);
            }
            const double count = samples.size();
            mean_gyro /= count;
            mean_accel /= count;
            const auto gyro_std = (square_gyro/count-mean_gyro.cwiseProduct(mean_gyro)).cwiseMax(0).cwiseSqrt().eval();
            const auto accel_std = (square_accel/count-mean_accel.cwiseProduct(mean_accel)).cwiseMax(0).cwiseSqrt().eval();
            const double span_s = double(samples.back().t_ns-samples.front().t_ns)*1e-9;
            stationary_bootstrap = span_s >= .4 && mean_gyro.norm() < .025 && gyro_std.maxCoeff() < .015 &&
                accel_std.maxCoeff() < .10 && std::abs(mean_accel.norm()-basalt::constants::g.norm()) < .75;
            if (stationary_bootstrap) {
                bias_gyro = mean_gyro;
                // At a stationary start only the radial accelerometer bias is
                // observable without a calibrated world attitude. Keep flash
                // calibration intact and initialize the estimator's bias state.
                bias_accel = mean_accel - basalt::constants::g.norm()*mean_accel.normalized();
            }
            std::cerr << "Initial IMU: stationary_candidate=" << stationary_bootstrap
                      << " span_s=" << span_s << " accel_norm=" << mean_accel.norm()
                      << " gyro_std=" << gyro_std.transpose() << " accel_std=" << accel_std.transpose() << '\n';
        }
        for (const auto& sample : samples) {
            if (sample.t_ns <= last_imu_ns) throw std::runtime_error("IMU timestamps do not increase");
            auto value = std::make_shared<basalt::ImuData<double>>();
            value->t_ns = sample.t_ns;
            value->gyro = Eigen::Vector3d(sample.gyro[0], sample.gyro[1], sample.gyro[2]);
            value->accel = Eigen::Vector3d(sample.accel[0], sample.accel[1], sample.accel[2]);
            if (!value->gyro.allFinite() || !value->accel.allFinite())
                throw std::runtime_error("non-finite IMU input");
            vio->imu_data_queue.push(value);
            if (!initialized) {
                if (stationary_bootstrap) {
                    Sophus::SE3d initial_pose;
                    initial_pose.setQuaternion(Eigen::Quaterniond::FromTwoVectors(mean_accel, Eigen::Vector3d::UnitZ()));
                    vio->initialize(header.image_ns, initial_pose, Eigen::Vector3d::Zero(), bias_gyro, bias_accel);
                } else {
                    vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
                }
                std::ofstream initialization(output_path + ".initialization.json");
                initialization << std::setprecision(17)
                    << "{\"stationary_imu_bootstrap\":" << (stationary_bootstrap ? "true" : "false")
                    << ",\"gyro_bias_rad_s\":";
                write_matrix(initialization, bias_gyro.transpose());
                initialization << ",\"accel_bias_m_s2\":";
                write_matrix(initialization, bias_accel.transpose());
                initialization << "}\n";
                initialized = true;
            }
            last_imu_ns = sample.t_ns;
            ++imu_count;
        }
        auto frame = std::make_shared<basalt::OpticalFlowInput>();
        if (!initialized) throw std::runtime_error("first image must be preceded by valid IMU samples");
        frame->t_ns = header.image_ns;
        frame->img_data.resize(2);
        for (size_t camera = 0; camera < 2; ++camera) {
            const int width = calibration.resolution[camera][0];
            const int height = calibration.resolution[camera][1];
            if (header.image_size[camera] != uint64_t(width)*height)
                throw std::runtime_error("wrong grayscale plane size for Basalt calibration");
            std::vector<uint8_t> image(header.image_size[camera]);
            read_bytes(reinterpret_cast<char*>(image.data()), image.size());
            auto managed = std::make_shared<basalt::ManagedImage<uint16_t>>(width, height);
            for (int y = 0; y < height; ++y) {
                const auto* row = image.data()+size_t(y)*width;
                for (int x = 0; x < width; ++x) (*managed)(x,y) = uint16_t(row[x]) << 8;
            }
            frame->img_data[camera].img = managed;
        }
        flow->input_queue.push(frame);
        last_image_ns = header.image_ns;
        ++image_count;
    }

    void finish() {
        if (finished) return;
        finished = true;
        flow->input_queue.push(nullptr);
        if (initialized) {
            vio->imu_data_queue.push(nullptr);
            vio->maybe_join();
            vio->drain_input_queues();
        } else {
            states.push(nullptr);
            maps.push(nullptr);
        }
        flow.reset();
        if (state_thread.joinable()) state_thread.join();
        if (map_thread.joinable()) map_thread.join();
        output.flush();
        std::cerr << "Basalt live finished: images=" << image_count << " imu=" << imu_count
                  << " poses=" << pose_count << " map_updates=" << map_count << '\n';
    }

    ~Engine() {
        if (!finished) {
            try { finish(); }
            catch (...) { std::_Exit(4); }
        }
    }
};
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) {
            std::cerr << "Usage: ego_basalt_live CALIBRATION_JSON CONFIG_JSON OUTPUT_JSONL IMAGE_IMU_SHIFT_NS [THREADS]\n";
            return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 2;
        }
        tbb::global_control threads(tbb::global_control::max_allowed_parallelism,
                                    argc == 6 ? std::stoul(argv[5]) : 6);
        Engine engine(argv[1],argv[2],argv[3],std::stoll(argv[4]));
        Header header{};
        while (read_header(header)) {
            if (engine.vio->finished) throw std::runtime_error("Basalt estimator stopped before end of input");
            engine.add(header);
        }
        engine.finish();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "ego_basalt_live: " << error.what() << '\n';
        return 1;
    }
}
