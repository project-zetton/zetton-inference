#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <string>

#include "zetton_inference/base/geometry/point_cloud.h"
#include "zetton_inference/base/object/object_supplement.h"

namespace zetton {
namespace inference {

struct alignas(16) Object {
 public:
  Object();
  void Reset();
  std::string ToString() const;

 public:
  /// \brief object id per frame, required
  int id = -1;

  /// \brief convex hull of the object, required
  PointCloud<PointD> polygon;

  // oriented boundingbox information
  /// \brief main direction of the object, required
  Eigen::Vector3f direction = Eigen::Vector3f(1, 0, 0);
  /// \brief the yaw angle, theta = 0.0 <=> direction(1, 0, 0),
  /// currently roll and pitch are not considered,
  /// make sure direction and theta are consistent, required
  float theta = 0.0f;
  /// \brief theta variance, required
  float theta_variance = 0.0f;

  // 2D boudningbox information
  /// \brief center of the boundingbox (cx, cy, cz), required
  Eigen::Vector3d center = Eigen::Vector3d(0, 0, 0);
  /// \brief covariance matrix of the center uncertainty, required
  Eigen::Matrix3f center_uncertainty;

  // 3D boudningbox information
  /// \brief brief size = [length, width, height] of boundingbox
  /// length is the size of the main direction, required
  Eigen::Vector3f size = Eigen::Vector3f(0, 0, 0);
  /// \brief size variance, required
  Eigen::Vector3f size_variance = Eigen::Vector3f(0, 0, 0);
  /// \brief anchor point, required
  Eigen::Vector3d anchor_point = Eigen::Vector3d(0, 0, 0);

  /// classification information
  /// \brief object type, required
  int type = -1;
  /// \brief probability for each type, required
  float type_prob;

  /// \brief existence confidence, required
  float confidence = 1.0f;

  // tracking information
  /// \brief track id, required
  int track_id = -1;
  /// \brief velocity of the object, required
  Eigen::Vector3f velocity = Eigen::Vector3f(0, 0, 0);
  /// \brief covariance matrix of the velocity uncertainty, required
  Eigen::Matrix3f velocity_uncertainty;
  /// \brief if the velocity estimation is converged, true by default
  bool velocity_converged = true;
  /// \brief velocity confidence, required
  float velocity_confidence = 1.0f;
  /// \brief acceleration of the object, required
  Eigen::Vector3f acceleration = Eigen::Vector3f(0, 0, 0);
  /// \brief covariance matrix of the acceleration uncertainty, required
  Eigen::Matrix3f acceleration_uncertainty;

  /// \brief age of the tracked object, required
  double tracking_time = 0.0;
  /// \brief timestamp of latest measurement, required
  double latest_tracked_time = 0.0;

  /// \brief motion state of the tracked object, required
  int motion_state = -1;

  /// \brief sensor-specific object supplements, optional
  CameraObjectSupplement camera_supplement;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

using ObjectPtr = std::shared_ptr<Object>;
using ObjectConstPtr = std::shared_ptr<const Object>;

}  // namespace inference
}  // namespace zetton
