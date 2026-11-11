cmake_minimum_required(VERSION 3.24)
project(basalt_local_comparison LANGUAGES C CXX)

# Build the original, unmodified Basalt sources with locally available support
# libraries. This avoids rebuilding all of vcpkg for a controlled comparison.
get_filename_component(REPO "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(ORIGINAL "${REPO}/third_party/basalt")
set(MONADO "${REPO}/third_party/basalt_monado")
set(BASALT_HEADERS "" CACHE PATH "Original basalt-headers at the pinned revision")
if(NOT EXISTS "${BASALT_HEADERS}/include/basalt/camera/generic_camera.hpp")
  message(FATAL_ERROR "Set BASALT_HEADERS to the original pinned basalt-headers checkout")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
add_compile_definitions(BASALT_INSTANTIATIONS_FLOAT EIGEN_DONT_PARALLELIZE FMT_DEPRECATED_OSTREAM=1)

find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d features2d imgcodecs)
find_package(TBB REQUIRED)
find_package(fmt REQUIRED)
find_package(Boost REQUIRED COMPONENTS date_time filesystem program_options regex)
find_package(BZip2 REQUIRED)

# Match the already built Monado binary's Eigen/Sophus/cereal support versions.
add_library(eigen INTERFACE)
target_include_directories(eigen SYSTEM INTERFACE "${MONADO}/thirdparty/basalt-headers/thirdparty/eigen")
add_library(Eigen3::Eigen ALIAS eigen)
set(Eigen3_FOUND TRUE)
add_library(sophus INTERFACE)
add_library(Sophus::Sophus ALIAS sophus)
target_include_directories(sophus SYSTEM INTERFACE "${MONADO}/thirdparty/basalt-headers/thirdparty/Sophus")
target_link_libraries(sophus INTERFACE eigen fmt::fmt)
add_library(original_headers INTERFACE)
target_include_directories(original_headers INTERFACE "${BASALT_HEADERS}/include")
target_include_directories(original_headers SYSTEM INTERFACE "${MONADO}/thirdparty/basalt-headers/thirdparty/cereal/include")
target_link_libraries(original_headers INTERFACE eigen sophus)
add_library(basalt::basalt-headers ALIAS original_headers)
include("${MONADO}/build/thirdparty/Pangolin/PangolinTargets.cmake")
set(PANGO pango_display pango_plot pango_geometry pango_glgeometry pango_scene pango_tools)
add_library(local_opengv STATIC IMPORTED)
set_target_properties(local_opengv PROPERTIES IMPORTED_LOCATION "${MONADO}/build/lib/libopengv.a"
  INTERFACE_INCLUDE_DIRECTORIES "${MONADO}/thirdparty/opengv/include")
add_library(local_rosbag STATIC IMPORTED)
set_target_properties(local_rosbag PROPERTIES IMPORTED_LOCATION "${MONADO}/build/thirdparty/ros/librosbag.a")
set(ROS "${MONADO}/thirdparty/ros")
target_include_directories(local_rosbag INTERFACE "${ROS}/include" "${ROS}/console_bridge/include"
  "${ROS}/roscpp_core/cpp_common/include" "${ROS}/roscpp_core/rostime/include"
  "${ROS}/roscpp_core/roscpp_serialization/include" "${ROS}/roscpp_core/roscpp_traits/include"
  "${ROS}/ros_comm/utilities/roslz4/include" "${ROS}/ros_comm/tools/rosbag_storage/include")
target_link_libraries(local_rosbag INTERFACE ${Boost_LIBRARIES} BZip2::BZip2 lz4)
add_library(basalt::opencv INTERFACE IMPORTED)
set_target_properties(basalt::opencv PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}" INTERFACE_LINK_LIBRARIES "${OpenCV_LIBS}")
add_subdirectory("${ORIGINAL}/thirdparty/apriltag" apriltag EXCLUDE_FROM_ALL)
target_link_libraries(apriltag PUBLIC original_headers)

# Read the upstream core source list rather than borrowing Monado's algorithm.
file(READ "${ORIGINAL}/CMakeLists.txt" UPSTREAM_CMAKE)
string(REGEX MATCHALL "\\$\\{CMAKE_CURRENT_SOURCE_DIR\\}/src/[A-Za-z0-9_/]+\\.cpp" CORE_SOURCES "${UPSTREAM_CMAKE}")
list(TRANSFORM CORE_SOURCES REPLACE "\\$\\{CMAKE_CURRENT_SOURCE_DIR\\}" "${ORIGINAL}")
list(REMOVE_DUPLICATES CORE_SOURCES)
add_library(basalt_original STATIC ${CORE_SOURCES})
target_include_directories(basalt_original PUBLIC "${ORIGINAL}/include")
target_include_directories(basalt_original SYSTEM PUBLIC "${MONADO}/thirdparty/magic_enum/include"
  "${MONADO}/thirdparty/CLI11/include" "${MONADO}/thirdparty/json")
target_link_libraries(basalt_original PUBLIC original_headers basalt::opencv TBB::tbb fmt::fmt
  local_rosbag local_opengv apriltag stdc++fs)
add_executable(basalt_vio "${ORIGINAL}/src/vio.cpp")
target_link_libraries(basalt_vio PRIVATE basalt_original ${PANGO})
set(TBB_ALIGNMENT_OVERLAY "" CACHE PATH "Optional oneTBB aligned-storage compatibility header overlay")
if(TBB_ALIGNMENT_OVERLAY)
  # Ubuntu's oneTBB value_node storage uses max_align_t instead of the value's
  # required alignment. Original Matches stores a 32-byte-aligned Eigen value.
  # Recompile only mapping with the local corrected header; keep VIO unchanged.
  add_library(mapping_nfr OBJECT "${ORIGINAL}/src/vi_estimator/nfr_mapper.cpp")
  target_include_directories(mapping_nfr BEFORE PRIVATE "${TBB_ALIGNMENT_OVERLAY}")
  target_link_libraries(mapping_nfr PRIVATE basalt_original)
  add_executable(basalt_mapper_export "${CMAKE_CURRENT_LIST_DIR}/mapper_export.cpp" $<TARGET_OBJECTS:mapping_nfr>)
  target_include_directories(basalt_mapper_export BEFORE PRIVATE "${TBB_ALIGNMENT_OVERLAY}")
else()
  add_executable(basalt_mapper_export "${CMAKE_CURRENT_LIST_DIR}/mapper_export.cpp")
endif()
target_link_libraries(basalt_mapper_export PRIVATE basalt_original ${PANGO})
