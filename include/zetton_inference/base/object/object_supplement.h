#pragma once

#include <Eigen/Core>
#include <memory>

#include "zetton_inference/base/geometry/box.h"

namespace zetton {
namespace inference {

struct alignas(16) CameraObjectSupplement {
  CameraObjectSupplement() { Reset(); }

  void Reset() {
    on_use = false;
    sensor_name.clear();

    box = BBox2D<float>();
    projected_box = BBox2D<float>();
    object_feature.clear();

    local_track_id = -1;
  }

  /// \brief valid only for on_use = true
  bool on_use = false;

  /// \brief camera sensor name
  std::string sensor_name;

  /// \brief 2D box
  BBox2D<float> box;

  /// \brief projected 2D box
  BBox2D<float> projected_box;

  /// \brief object visual feature map
  std::vector<float> object_feature;

  /// \brief local track id
  int local_track_id = -1;
};

using CameraObjectSupplementPtr = std::shared_ptr<CameraObjectSupplement>;
using CameraObjectSupplementConstPtr =
    std::shared_ptr<const CameraObjectSupplement>;

}  // namespace inference
}  // namespace zetton
