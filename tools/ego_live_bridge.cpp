// Copyright (c) 2026 EGO Hand System contributors.
// Orbbec SDK calls follow the vendor's MIT-licensed EGO sample.

#include <libobsensor/ObSensor.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

#pragma pack(push, 1)
struct StereoPacketHeader {
    char     magic[4];
    uint16_t version;
    uint16_t headerSize;
    uint32_t leftSize;
    uint32_t rightSize;
    uint32_t width;
    uint32_t height;
    uint64_t leftTimestampUs;
    uint64_t rightTimestampUs;
    uint64_t leftIndex;
    uint64_t rightIndex;
    uint64_t hostMonotonicNs;
};
#pragma pack(pop)

struct Options {
    std::string sdkConfig;
    std::string calibrationOut;
    uint32_t    width      = 1600;
    uint32_t    height     = 1300;
    uint32_t    fps        = 30;
    uint64_t    maxFrames  = 0;
    uint32_t    timeoutMs  = 100;
    bool        calibrationOnly = false;
};

struct EncodedStereoPacket {
    StereoPacketHeader header{};
    std::vector<uint8_t> left;
    std::vector<uint8_t> right;
};

uint64_t parseUnsigned(const char *value, const char *name) {
    try {
        size_t position = 0;
        auto result = std::stoull(value, &position);
        if(position != std::strlen(value)) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    }
    catch(const std::exception &) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
    }
}

Options parseArgs(int argc, char **argv) {
    Options options;
    for(int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto requireValue = [&](const char *name) -> const char * {
            if(index + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++index];
        };
        if(arg == "--sdk-config") {
            options.sdkConfig = requireValue("--sdk-config");
        }
        else if(arg == "--calibration-out") {
            options.calibrationOut = requireValue("--calibration-out");
        }
        else if(arg == "--width") {
            options.width = static_cast<uint32_t>(parseUnsigned(requireValue("--width"), "width"));
        }
        else if(arg == "--height") {
            options.height = static_cast<uint32_t>(parseUnsigned(requireValue("--height"), "height"));
        }
        else if(arg == "--fps") {
            options.fps = static_cast<uint32_t>(parseUnsigned(requireValue("--fps"), "fps"));
        }
        else if(arg == "--max-frames") {
            options.maxFrames = parseUnsigned(requireValue("--max-frames"), "max-frames");
        }
        else if(arg == "--timeout-ms") {
            options.timeoutMs = static_cast<uint32_t>(parseUnsigned(requireValue("--timeout-ms"), "timeout-ms"));
        }
        else if(arg == "--calibration-only") {
            options.calibrationOnly = true;
        }
        else if(arg == "--help" || arg == "-h") {
            std::cerr
                << "Usage: ego_live_bridge [--sdk-config FILE] [--calibration-out FILE]\n"
                << "       [--calibration-only] [--width 1600] [--height 1300] [--fps 30]\n"
                << "       [--max-frames N] [--timeout-ms 100]\n";
            std::exit(EXIT_SUCCESS);
        }
        else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if(options.width == 0 || options.height == 0 || options.fps == 0 || options.timeoutMs == 0) {
        throw std::invalid_argument("width, height, fps and timeout must be positive");
    }
    if(options.calibrationOnly && options.calibrationOut.empty()) {
        throw std::invalid_argument("--calibration-only requires --calibration-out");
    }
    return options;
}

bool hasSensor(const std::shared_ptr<ob::Device> &device, OBSensorType type) {
    auto sensors = device->getSensorList();
    for(uint32_t index = 0; index < sensors->getCount(); ++index) {
        if(sensors->getSensorType(index) == type) {
            return true;
        }
    }
    return false;
}

void saveCalibration(const std::shared_ptr<ob::Device> &device, const std::string &path) {
    if(path.empty()) {
        return;
    }
    std::vector<uint8_t> bytes;
    bool completed = false;
    int transferState = DATA_TRAN_STAT_TRANSFERRING;
    device->getRawData(OB_RAW_DATA_ALIGN_CALIB_YAML, [&](OBDataTranState state, OBDataChunk *chunk) {
        transferState = static_cast<int>(state);
        if(chunk != nullptr && chunk->data != nullptr && chunk->size > 0) {
            const size_t needed = static_cast<size_t>(chunk->offset) + chunk->size;
            if(bytes.size() < needed) {
                bytes.resize(needed);
            }
            std::copy(chunk->data, chunk->data + chunk->size, bytes.begin() + chunk->offset);
        }
        if(state == DATA_TRAN_STAT_DONE || state == DATA_TRAN_STAT_VERIFY_DONE) {
            completed = true;
        }
    });
    if(!completed || bytes.empty()) {
        throw std::runtime_error("failed to read EGO calibration YAML, transfer state=" + std::to_string(transferState));
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if(!stream) {
        throw std::runtime_error("cannot create calibration file: " + path);
    }
    stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if(!stream) {
        throw std::runtime_error("failed to write calibration file: " + path);
    }
    std::cerr << "Calibration YAML: " << path << " (" << bytes.size() << " bytes)\n";
}

std::shared_ptr<ob::Frame> frameOfType(const std::shared_ptr<ob::FrameSet> &frames, OBFrameType type) {
    auto frame = frames->getFrame(type);
    if(frame) {
        return frame;
    }
    for(uint32_t index = 0; index < frames->getCount(); ++index) {
        auto candidate = frames->getFrameByIndex(index);
        if(candidate && candidate->getType() == type) {
            return candidate;
        }
    }
    return nullptr;
}

std::shared_ptr<EncodedStereoPacket> copyPacket(
    const std::shared_ptr<ob::Frame> &left,
    const std::shared_ptr<ob::Frame> &right) {
    auto leftVideo = left->as<ob::VideoFrame>();
    auto rightVideo = right->as<ob::VideoFrame>();
    if(left->getFormat() != OB_FORMAT_MJPG || right->getFormat() != OB_FORMAT_MJPG) {
        throw std::runtime_error("EGO live bridge requires MJPG left/right frames");
    }
    if(leftVideo->getWidth() != rightVideo->getWidth() || leftVideo->getHeight() != rightVideo->getHeight()) {
        throw std::runtime_error("left/right frame dimensions disagree");
    }
    if(left->getDataSize() > UINT32_MAX || right->getDataSize() > UINT32_MAX) {
        throw std::runtime_error("compressed frame is too large");
    }

    auto packet = std::make_shared<EncodedStereoPacket>();
    std::memcpy(packet->header.magic, "EGO1", 4);
    packet->header.version          = 1;
    packet->header.headerSize       = sizeof(packet->header);
    packet->header.leftSize         = static_cast<uint32_t>(left->getDataSize());
    packet->header.rightSize        = static_cast<uint32_t>(right->getDataSize());
    packet->header.width            = leftVideo->getWidth();
    packet->header.height           = leftVideo->getHeight();
    packet->header.leftTimestampUs  = left->getTimeStampUs();
    packet->header.rightTimestampUs = right->getTimeStampUs();
    packet->header.leftIndex        = left->getIndex();
    packet->header.rightIndex       = right->getIndex();
    packet->header.hostMonotonicNs  = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    const auto *leftData = static_cast<const uint8_t *>(left->getData());
    const auto *rightData = static_cast<const uint8_t *>(right->getData());
    packet->left.assign(leftData, leftData + left->getDataSize());
    packet->right.assign(rightData, rightData + right->getDataSize());
    return packet;
}

void writePacket(const EncodedStereoPacket &packet) {
    std::cout.write(reinterpret_cast<const char *>(&packet.header), sizeof(packet.header));
    std::cout.write(reinterpret_cast<const char *>(packet.left.data()),
                    static_cast<std::streamsize>(packet.left.size()));
    std::cout.write(reinterpret_cast<const char *>(packet.right.data()),
                    static_cast<std::streamsize>(packet.right.size()));
    std::cout.flush();
    if(!std::cout) {
        throw std::runtime_error("live bridge output pipe was closed");
    }
}

}  // namespace

int main(int argc, char **argv) try {
    const Options options = parseArgs(argc, argv);
    ob::Context::setLoggerToConsole(OB_LOG_SEVERITY_OFF);
    ob::Context context(options.sdkConfig.empty() ? "" : options.sdkConfig.c_str());
#if defined(__linux__) || defined(__ANDROID__)
    context.setUvcBackendType(OB_UVC_BACKEND_TYPE_LIBUVC);
#endif

    ob::Pipeline pipeline;
    auto device = pipeline.getDevice();
    auto info = device->getDeviceInfo();
    std::cerr << "Device: " << info->getName() << " serial=" << info->getSerialNumber()
              << " vid=0x" << std::hex << info->getVid() << " pid=0x" << info->getPid() << std::dec
              << " connection=" << info->getConnectionType() << "\n";
    if(std::string(info->getConnectionType()).find("USB2") != std::string::npos) {
        std::cerr << "WARNING: EGO is connected through USB2; dual 1600x1300 MJPG "
                     "throughput may be limited. Use a direct USB3 port/cable when available.\n";
    }
    if(info->getVid() != 0x2bc5 || info->getPid() != 0x1201) {
        throw std::runtime_error("connected device is not the supported EGO 2bc5:1201");
    }
    if(!hasSensor(device, OB_SENSOR_COLOR_LEFT) || !hasSensor(device, OB_SENSOR_COLOR_RIGHT)) {
        throw std::runtime_error("EGO does not expose both COLOR_LEFT and COLOR_RIGHT sensors");
    }

    saveCalibration(device, options.calibrationOut);
    if(options.calibrationOnly) {
        return EXIT_SUCCESS;
    }

    auto config = std::make_shared<ob::Config>();
    config->enableVideoStream(OB_STREAM_COLOR_LEFT, options.width, options.height, options.fps, OB_FORMAT_MJPG);
    config->enableVideoStream(OB_STREAM_COLOR_RIGHT, options.width, options.height, options.fps, OB_FORMAT_MJPG);
    pipeline.start(config);
    std::cerr << "Streaming EGO stereo MJPG " << options.width << "x" << options.height
              << " @ " << options.fps << " FPS\n";

    std::mutex packetMutex;
    std::condition_variable packetReady;
    std::shared_ptr<EncodedStereoPacket> latestPacket;
    uint64_t generation = 0;
    bool stopping = false;
    std::exception_ptr writerError;
    std::atomic<uint64_t> emitted{0};
    std::thread writer([&] {
        uint64_t writtenGeneration = 0;
        try {
            while(true) {
                std::shared_ptr<EncodedStereoPacket> packet;
                uint64_t selectedGeneration = 0;
                {
                    std::unique_lock<std::mutex> lock(packetMutex);
                    packetReady.wait(lock, [&] {
                        return stopping || generation > writtenGeneration;
                    });
                    if(generation <= writtenGeneration && stopping) {
                        break;
                    }
                    packet = latestPacket;
                    selectedGeneration = generation;
                }
                writePacket(*packet);
                writtenGeneration = selectedGeneration;
                ++emitted;
            }
        }
        catch(...) {
            std::lock_guard<std::mutex> lock(packetMutex);
            writerError = std::current_exception();
            stopping = true;
            packetReady.notify_all();
        }
    });

    uint64_t captured = 0;
    uint64_t missing = 0;
    while(options.maxFrames == 0 || captured < options.maxFrames) {
        {
            std::lock_guard<std::mutex> lock(packetMutex);
            if(stopping) {
                break;
            }
        }
        auto frames = pipeline.waitForFrameset(options.timeoutMs);
        if(!frames) {
            ++missing;
            if(missing % 50 == 0) {
                std::cerr << "Waiting for complete EGO stereo frameset...\n";
            }
            continue;
        }
        auto left = frameOfType(frames, OB_FRAME_COLOR_LEFT);
        auto right = frameOfType(frames, OB_FRAME_COLOR_RIGHT);
        if(!left || !right) {
            ++missing;
            continue;
        }
        auto packet = copyPacket(left, right);
        {
            std::lock_guard<std::mutex> lock(packetMutex);
            latestPacket = std::move(packet);
            ++generation;
        }
        ++captured;
        packetReady.notify_one();
    }
    pipeline.stop();
    {
        std::lock_guard<std::mutex> lock(packetMutex);
        stopping = true;
    }
    packetReady.notify_all();
    writer.join();
    if(writerError) {
        std::rethrow_exception(writerError);
    }
    std::cerr << "Stereo packets captured/emitted/dropped: " << captured << "/"
              << emitted.load() << "/" << (captured - emitted.load()) << "\n";
    return EXIT_SUCCESS;
}
catch(const ob::Error &error) {
    std::cerr << "Orbbec error: function=" << error.getFunction() << " args=" << error.getArgs()
              << " message=" << error.what() << " status=" << error.getStatus() << "\n";
    return EXIT_FAILURE;
}
catch(const std::exception &error) {
    std::cerr << "error: " << error.what() << "\n";
    return EXIT_FAILURE;
}
