// Headless export using Basalt's existing NFR mapping implementation.
#include <basalt/io/marg_data_io.h>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/vio_config.h>
#include <basalt/vi_estimator/nfr_mapper.h>
#include <tbb/global_control.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_set>

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: basalt_mapper_export calibration.json config.json marg_data output_dir\n";
    return 2;
  }
  namespace fs = std::filesystem;
  const fs::path output(argv[4]);
  if (fs::exists(output)) {
    std::cerr << "Choose a fresh output directory: " << output << '\n';
    return 2;
  }
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 2);
  basalt::Calibration<double> calib;
  {
    std::ifstream stream(argv[1]);
    cereal::JSONInputArchive archive(stream);
    archive(calib);
  }
  basalt::VioConfig config;
  config.load(argv[2]);
  basalt::NfrMapper mapper(calib, config);
  basalt::MargDataLoader loader;
  tbb::concurrent_bounded_queue<basalt::MargData::Ptr> queue;
  loader.out_marg_queue = &queue;
  loader.start(argv[3]);
  size_t snapshots = 0;
  while (true) {
    basalt::MargData::Ptr data;
    queue.pop(data);
    if (!data) break;
    mapper.addMargData(data);
    ++snapshots;
  }
  if (mapper.getFramePoses().empty()) {
    std::cerr << "No usable keyframes in marginalization cache\n";
    return 1;
  }
  std::cout << "Loaded " << snapshots << " snapshots, " << mapper.getFramePoses().size() << " keyframes\n";
  mapper.detect_keypoints();
  mapper.match_stereo();
  mapper.match_all();
  mapper.build_tracks();
  mapper.setup_opt();
  mapper.optimize(10);
  mapper.filterOutliers(3.0, 4);
  mapper.optimize(10);
  Eigen::aligned_vector<Eigen::Vector3d> points;
  std::vector<int> ids;
  mapper.get_current_points(points, ids);
  // Some upstream host lists repeat a landmark for multiple observations.
  // A point-cloud vertex must represent a unique landmark, not an observation.
  std::unordered_set<int> seen;
  std::vector<size_t> unique_indices;
  for (size_t i = 0; i < points.size(); ++i) {
    if (points[i].allFinite() && seen.insert(ids.at(i)).second) unique_indices.push_back(i);
  }
  const size_t valid = unique_indices.size();
  if (valid == 0) {
    std::cerr << "Mapping produced no finite landmarks\n";
    return 1;
  }
  fs::create_directories(output);
  std::ofstream ply(output / "sparse_map.ply"), csv(output / "sparse_landmarks.csv");
  ply << "ply\nformat ascii 1.0\ncomment Basalt NFR sparse landmarks; world frame; metres\n"
      << "element vertex " << valid << "\nproperty double x\nproperty double y\nproperty double z\nend_header\n";
  csv << "landmark_id,x_m,y_m,z_m\n";
  ply << std::setprecision(16);
  csv << std::setprecision(16);
  for (size_t i : unique_indices) {
    const auto& p = points[i];
    if (!p.allFinite()) continue;
    ply << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    csv << ids.at(i) << ',' << p.x() << ',' << p.y() << ',' << p.z() << '\n';
  }
  for (int head = 0; head < 2; ++head) {
    std::ofstream trajectory(output / (head ? "keyframe_headring_poses.csv" : "keyframe_imu_poses.csv"));
    trajectory << "timestamp_ns,x_m,y_m,z_m,qw,qx,qy,qz\n" << std::setprecision(16);
    for (const auto& [timestamp, state] : mapper.getFramePoses()) {
      Sophus::SE3d pose = state.getPose();
      if (head) pose = pose * calib.T_i_c.at(0);
      const auto& p = pose.translation();
      const auto q = pose.unit_quaternion();
      trajectory << timestamp << ',' << p.x() << ',' << p.y() << ',' << p.z() << ','
                 << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << '\n';
    }
  }
  std::ofstream summary(output / "summary.json");
  summary << "{\n  \"landmarks\": " << valid << ",\n  \"keyframes\": " << mapper.getFramePoses().size()
          << ",\n  \"marginalization_snapshots\": " << snapshots
          << ",\n  \"map_type\": \"Basalt NFR optimized sparse feature landmarks\","
          << "\n  \"units\": \"metres\",\n  \"headring_reference\": \"factory cam_0 optical frame\"\n}\n";
  std::cout << "Exported " << valid << " sparse landmarks to " << output << '\n';
  return 0;
}
