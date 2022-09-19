#include "zetton_inference/base/object/object.h"

#include "fmt/format.h"

namespace zetton {
namespace inference {

Object::Object() {
  center_uncertainty << 0.0f, 0, 0, 0, 0.0f, 0, 0, 0, 0.0f;
  velocity_uncertainty << 0.0f, 0, 0, 0, 0.0f, 0, 0, 0, 0.0f;
  acceleration_uncertainty << 0.0f, 0, 0, 0, 0.0f, 0, 0, 0, 0.0f;
}

void Object::Reset() {
  id = -1;

  polygon.clear();

  // oriented boundingbox information
  direction = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
  theta = 0.0f;
  theta_variance = 0.0f;

  // 2D boudningbox information
  center = Eigen::Vector3d(0.0, 0.0, 0.0);
  center_uncertainty << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f;

  // 3D boudningbox information
  size = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
  size_variance = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
  anchor_point = Eigen::Vector3d(0.0, 0.0, 0.0);

  /// classification information
  type = -1;
  type_prob = 0.0;
  confidence = 1.0f;

  // tracking information
  track_id = -1;
  velocity = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
  velocity_uncertainty << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f;
  velocity_converged = true;
  velocity_confidence = 1.0;
  acceleration = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
  acceleration_uncertainty << 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f;

  tracking_time = 0.0;
  latest_tracked_time = 0.0;

  motion_state = -1;
}

std::string Object::ToString() const {
  return fmt::format(
      "Object [id: {}], track_id: {}, direction: ({:.2f}, {:.2f}, {:.2f}), "
      "theta: {:.2f}, theta_variance: {:.2f}, center: ({:.2f}, {:.2f}, "
      "{:.2f}), center_uncertainty: ({:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, "
      "{:.2f}, {:.2f}, {:.2f}, {:.2f}), size: ({:.2f}, {:.2f}, {:.2f}), "
      "size_variance: ({:.2f}, {:.2f}, {:.2f}), anchor_point: ({:.2f}, {:.2f}, "
      "{:.2f}), type: {}, confidence: {:.2f}, velocity: ({:.2f}, {:.2f}, "
      "{:.2f}), velocity_confidence: {:.2f}, acceleration: ({:.2f}, {:.2f}, "
      "{:.2f}), tracking_time: {:.2f}, latest_tracked_time: {:.2f}",
      id, track_id, direction[0], direction[1], direction[2], theta,
      theta_variance, center[0], center[1], center[2], center_uncertainty(0, 0),
      center_uncertainty(0, 1), center_uncertainty(0, 2),
      center_uncertainty(1, 0), center_uncertainty(1, 1),
      center_uncertainty(1, 2), center_uncertainty(2, 0),
      center_uncertainty(2, 1), center_uncertainty(2, 2), size[0], size[1],
      size[2], size_variance[0], size_variance[1], size_variance[2],
      anchor_point[0], anchor_point[1], anchor_point[2], type, confidence,
      velocity[0], velocity[1], velocity[2], velocity_confidence,
      acceleration[0], acceleration[1], acceleration[2], tracking_time,
      latest_tracked_time);
}
}  // namespace inference
}  // namespace zetton
