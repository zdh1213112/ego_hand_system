cmake_minimum_required(VERSION 3.16)

project(ego_hand_system VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio calib3d aruco)
find_package(yaml-cpp REQUIRED)

option(EGO_HAND_BUILD_LIVE "Build the Orbbec EGO real-time bridge" ON)
option(EGO_HAND_BUILD_BASALT_LIVE "Build the continuous Basalt stereo/IMU adapter" OFF)
set(ORBBEC_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/orbbec_sdk" CACHE PATH
    "Path to the bundled OrbbecSDK runtime")

add_library(ego_hand_core
    src/calibration.cpp
    src/session.cpp
)

target_include_directories(ego_hand_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ego_hand_core
    PUBLIC
        ${OpenCV_LIBS}
        yaml-cpp
)

target_compile_options(ego_hand_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(ego_session_inspect tools/session_inspect.cpp)
target_link_libraries(ego_session_inspect PRIVATE ego_hand_core)
target_compile_options(ego_session_inspect PRIVATE -Wall -Wextra -Wpedantic)

add_executable(ego_epipolar_validate tools/epipolar_validate.cpp)
target_link_libraries(ego_epipolar_validate PRIVATE ego_hand_core)
target_compile_options(ego_epipolar_validate PRIVATE -Wall -Wextra -Wpedantic)

if(EGO_HAND_BUILD_LIVE)
    if(EXISTS "${ORBBEC_SDK_ROOT}/lib/OrbbecSDKConfig.cmake")
        find_package(OrbbecSDK CONFIG REQUIRED
            PATHS "${ORBBEC_SDK_ROOT}/lib"
            NO_DEFAULT_PATH
        )
        add_executable(ego_live_bridge tools/ego_live_bridge.cpp)
        target_link_libraries(ego_live_bridge PRIVATE ob::OrbbecSDK)
        target_compile_options(ego_live_bridge PRIVATE -Wall -Wextra -Wpedantic)
        set_target_properties(ego_live_bridge PROPERTIES
            BUILD_RPATH "${ORBBEC_SDK_ROOT}/lib"
            INSTALL_RPATH "$ORIGIN/../third_party/orbbec_sdk/lib"
        )
    else()
        message(WARNING
            "Orbbec SDK not found at ${ORBBEC_SDK_ROOT}; ego_live_bridge will not be built. "
            "Copy the Linux x86_64 SDK to third_party/orbbec_sdk or set ORBBEC_SDK_ROOT."
        )
    endif()
endif()

if(EGO_HAND_BUILD_BASALT_LIVE)
    set(BASALT_LIVE_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/basalt_live_deps" CACHE PATH
        "Pinned header-only dependencies installed by scripts/install_basalt_live_headers.py")
    set(BASALT_RUNTIME "${CMAKE_CURRENT_SOURCE_DIR}/third_party/basalt_runtime" CACHE PATH
        "Verified Basalt 0.1.7 runtime")
    if(NOT EXISTS "${BASALT_LIVE_DEPS}/basalt-headers/include/basalt/calibration/calibration.hpp"
       OR NOT EXISTS "${BASALT_RUNTIME}/lib/libbasalt.so")
        message(FATAL_ERROR "Install the Basalt runtime and run scripts/install_basalt_live_headers.py first")
    endif()
    find_package(Eigen3 REQUIRED)
    find_package(TBB REQUIRED)
    find_package(fmt REQUIRED)
    find_package(Threads REQUIRED)
    add_executable(ego_basalt_live tools/ego_basalt_live.cpp)
    target_include_directories(ego_basalt_live SYSTEM PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/basalt/include"
        "${BASALT_LIVE_DEPS}/basalt-headers/include"
        "${BASALT_LIVE_DEPS}/Sophus"
        "${BASALT_LIVE_DEPS}/cereal/include")
    target_link_libraries(ego_basalt_live PRIVATE
        "${BASALT_RUNTIME}/lib/libbasalt.so" Eigen3::Eigen TBB::tbb fmt::fmt
        Threads::Threads)
    target_compile_options(ego_basalt_live PRIVATE -Wall -Wextra -Wpedantic -march=native)
    set_target_properties(ego_basalt_live PROPERTIES BUILD_RPATH "${BASALT_RUNTIME}/lib"
        INSTALL_RPATH "$ORIGIN/../third_party/basalt_runtime/lib")
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(ego_core_tests tests/core_tests.cpp)
    target_link_libraries(ego_core_tests PRIVATE ego_hand_core)
    add_test(NAME ego_core_tests COMMAND ego_core_tests)
endif()
