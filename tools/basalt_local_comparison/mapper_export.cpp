// Identical headless NFR workflow for both Basalt branches. No algorithm changes.
#include <basalt/io/marg_data_io.h>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/vio_config.h>
#include <basalt/vi_estimator/nfr_mapper.h>
#include <basalt/camera/stereographic_param.hpp>
#include <tbb/global_control.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: mapper_export calibration.json config.json marg_data output_dir\n";
    return 2;
  }
  namespace fs = std::filesystem;
  const fs::path output(argv[4]);
  if (fs::exists(output)) return 2;
  tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 2);
  basalt::Calibration<double> calib;
  { std::ifstream f(argv[1]); cereal::JSONInputArchive ar(f); ar(calib); }
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
  if (mapper.getFramePoses().empty()) return 1;
  mapper.detect_keypoints();
  mapper.match_stereo();
  mapper.match_all();
  mapper.build_tracks();
  mapper.setup_opt();
  mapper.optimize(10);
  mapper.filterOutliers(3.0, 4);
  mapper.optimize(10);

  fs::create_directories(output);
  std::ofstream landmarks(output / "sparse_landmarks.csv");
  std::ofstream observations(output / "map_observations.csv");
  landmarks << "landmark_id,x_m,y_m,z_m,observation_count\n" << std::setprecision(16);
  observations << "landmark_id,timestamp_ns,cam_id,u_px,v_px,predicted_u_px,predicted_v_px,error_px\n" << std::setprecision(16);
  Eigen::aligned_vector<Eigen::Vector3d> points;
  std::vector<double> errors;
  std::vector<size_t> observations_per_camera(calib.intrinsics.size(), 0);
  size_t invalid_projections = 0;
  // Iterate the actual landmark database once. Original Basalt's legacy
  // get_current_points() fills every output ID with 1; deduplicating those IDs
  // would incorrectly turn a full original map into a one-point map.
  for (const auto& [id, lm] : mapper.lmdb.getLandmarks()) {
    const auto& host = lm.host_kf_id;
    const Sophus::SE3d T_w_c = mapper.getPoseStateWithLin(host.frame_id).getPose() * calib.T_i_c.at(host.cam_id);
    Eigen::Vector4d point = basalt::StereographicParam<double>::unproject(lm.direction);
    point[3] = lm.inv_dist;
    Eigen::Vector4d world = T_w_c.matrix() * point;
    if (!world.allFinite() || std::abs(world[3]) < 1e-12) continue;
    Eigen::Vector3d xyz = world.head<3>() / world[3];
    points.push_back(xyz);
    landmarks << id << ',' << xyz.x() << ',' << xyz.y() << ',' << xyz.z() << ',' << lm.obs.size() << '\n';
    for (const auto& [tcid, uv] : lm.obs) {
      const Sophus::SE3d target = mapper.getPoseStateWithLin(tcid.frame_id).getPose() * calib.T_i_c.at(tcid.cam_id);
      Eigen::Vector4d camera_point = target.inverse().matrix() * world;
      Eigen::Vector2d predicted;
      if (!calib.intrinsics.at(tcid.cam_id).project(camera_point, predicted) || !predicted.allFinite()) {
        ++invalid_projections;
        continue;
      }
      double error = (predicted - uv).norm();
      errors.push_back(error);
      ++observations_per_camera.at(tcid.cam_id);
      observations << id << ',' << tcid.frame_id << ',' << tcid.cam_id << ',' << uv.x() << ',' << uv.y()
                   << ',' << predicted.x() << ',' << predicted.y() << ',' << error << '\n';
    }
  }
  if (points.empty()) return 1;
  std::ofstream ply(output / "sparse_map.ply");
  ply << "ply\nformat ascii 1.0\ncomment NFR unique database landmarks; world frame; metres\nelement vertex "
      << points.size() << "\nproperty double x\nproperty double y\nproperty double z\nend_header\n" << std::setprecision(16);
  for (const auto& p : points) ply << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
  for (int camera = 0; camera <= 1; ++camera) {
    std::ofstream poses(output / (camera ? "keyframe_cam0_poses.csv" : "keyframe_imu_poses.csv"));
    poses << "timestamp_ns,x_m,y_m,z_m,qw,qx,qy,qz\n" << std::setprecision(16);
    for (const auto& [timestamp, state] : mapper.getFramePoses()) {
      Sophus::SE3d pose = state.getPose();
      if (camera) pose = pose * calib.T_i_c.at(0);
      const auto& p = pose.translation();
      const auto q = pose.unit_quaternion();
      poses << timestamp << ',' << p.x() << ',' << p.y() << ',' << p.z() << ','
            << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << '\n';
    }
  }
  double sum_sq = 0;
  for (double e : errors) sum_sq += e * e;
  std::sort(errors.begin(), errors.end());
  auto percentile = [&](double p) { return errors.empty() ? 0 : errors.at(size_t(p * (errors.size() - 1))); };
  std::ofstream summary(output / "summary.json");
  summary << std::setprecision(16) << "{\n  \"landmarks\": " << points.size()
          << ",\n  \"keyframes\": " << mapper.getFramePoses().size()
          << ",\n  \"marginalization_snapshots\": " << snapshots
          << ",\n  \"valid_reprojection_observations\": " << errors.size()
          << ",\n  \"invalid_projection_observations\": " << invalid_projections
          << ",\n  \"reprojection_rmse_px\": " << (errors.empty() ? 0 : std::sqrt(sum_sq / errors.size()))
          << ",\n  \"reprojection_median_px\": " << percentile(.5)
          << ",\n  \"reprojection_p95_px\": " << percentile(.95)
          << ",\n  \"observations_per_camera\": [";
  for (size_t i = 0; i < observations_per_camera.size(); ++i) {
    if (i) summary << ',';
    summary << observations_per_camera[i];
  }
  summary << "],\n  \"map_type\": \"NFR optimized sparse landmarks, 10 iterations then outlier filtering then 10 iterations\","
          << "\n  \"reprojection_scope\": \"in-sample residual after outlier filtering; not external map accuracy\","
          << "\n  \"units\": \"metres\"\n}\n";
  std::cout << "Exported " << points.size() << " landmarks and " << errors.size() << " observations\n";
  return 0;
}
